# MoE Kernel Fusion — Research Summary

Target: Fused SwiGLU + block-FP8 quant epilogue for MoE grouped GEMM, integrated into vLLM.

---

## The Opportunity

### Why MoE MLP is the bottleneck
In dense models, attention and MLP runtime roughly follow the parameter split (~1:2). In MoE this shifts: attention stays dense and well-optimized, but the MoE MLP runs hundreds of small per-expert GEMMs with poor arithmetic intensity. With M=1024 (continuous batching), 512 experts, and top-10 routing, each expert sees only 5-50 tokens. Per-expert GEMMs are `[5-50, 4096] × [4096, 2048]` — completely memory-bound. **MLP takes a disproportionately larger share of runtime in MoE than in dense models.**

### What "grouped GEMM" actually is
Not a single large GEMM. Each expert has its own weight matrix, its own token subset, its own M dimension. A grouped GEMM batches these independent GEMMs into one kernel launch to avoid launch overhead — different thread blocks handle different experts' tiles. The number of GEMMs equals the number of experts that received ≥1 token (verified via PyTorch blog on MoE kernel design). For Qwen3.5 with M=1024: ~200-400 active experts per layer → ~400-800 unique GEMMs per MoE layer (gate_up + down_proj), plus 2 for the shared expert. Across 60 layers: ~24,000-48,000 unique GEMMs per forward pass just for MLP.

### The fusion target
Between GEMM1 (gate_up) and GEMM2 (down_proj), there are separate kernels for SiLU*mul activation and FP8 quantization. Fusing these into GEMM1's epilogue eliminates those kernel launches. The epilogue runs per-CTA after each tile and doesn't care which expert produced the accumulator — same code works for all experts.

Down_proj (GEMM2) input dequant is a mainloop concern, not an epilogue. Not a fusion target.

---

## The Gap

### Competitive landscape

| | SM90 (H100) | SM100 (B200) |
|---|---|---|
| **cuDNN** | ❌ No MoE fusion | ✅ Full GEMM1+SwiGLU+FP8 fusion, but MXFP8 only, m_aligned=256, not used by vLLM |
| **FlashInfer CUTLASS** | ✅ Default FP8 MoE in vLLM, **no epilogue fusion (confirmed from source)** | ❌ Not the default on SM100 |
| **FlashInfer TRT-LLM gen** | ❌ Not used on SM90 | ✅ Default FP8/FP4 MoE in vLLM, **epilogue fusion unknown (black box)** |
| **vLLM Triton** | ✅ Fallback, no epilogue fusion | ✅ Fallback, not optimal |
| **Ours (target)** | ✅ Fused epilogue, vLLM-native block-FP8 | ⚠️ Depends on whether TRT-LLM gen already fuses |

### Why cuDNN doesn't solve this
NVIDIA's cuDNN ships `GroupedGemmSwigluSm100` — GEMM1 + SwiGLU + FP8 output quant in one kernel. But:
- **SM100 only** — no H100 support
- **MXFP8 format** (`float8_e8m0fnu` scales, block-32) — different from vLLM's DeepSeek-style block-128 FP8
- **m_aligned=256** — each expert's M padded to 256. With 5-50 tokens per expert, massive compute waste
- **cuDNN graph API** — doesn't integrate with vLLM's piecewise CUDA graph capture
- **NVIDIA-only** — vLLM also targets AMD

vLLM will not adopt cuDNN for MoE. Relevant only for TensorRT-LLM.

### FlashInfer: two backends, not one

FlashInfer provides two MoE backends, both accessible through vLLM:

**1. CUTLASS backend** (`cutlass_fused_moe`) — SM89/SM90 (H100)
- JIT-compiled CUTLASS grouped GEMMs with auto-tuning
- GEMM1 and GEMM2 are separate kernel launches with independent tactic tuning
- SwiGLU activation and FP8 quant between GEMMs are separate kernels
- Supports DeepSeek-style block-128 FP8 (`use_deepseek_fp8_block_scale=True`, SM90 only)
- **Gap confirmed from source:** `gemm_idx_for_tuning = 1` then `= 2` proves separate launches. `activation_type` handled as a separate step between GEMMs.

**2. TRT-LLM gen backend** (`trtllm_*_moe`) — SM100+ (B200/GB200)
- Wraps TensorRT-LLM's compiled C++ MoE implementation
- Two separate tactic selections (one per GEMM), same pattern as CUTLASS backend
- Recommended for FP8/FP4 on Blackwell: `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`
- Supports MXFP8, MXFP4, DeepSeek FP8, BF16
- **Gap uncertain:** C++ internals are opaque. Two-tactic pattern proves GEMMs are separate launches, but doesn't reveal whether SwiGLU + quant is fused into GEMM1's epilogue or runs as a separate kernel. May use cuDNN's fused kernel internally.

### vLLM MoE backend summary
| GPU | Backend | FP8 Format | SwiGLU+Quant Fused? |
|-----|---------|-----------|---------------------|
| H100 (SM90) | FlashInfer CUTLASS | DeepSeek block-128 FP8 | ❌ Confirmed separate |
| B200 (SM100) | FlashInfer TRT-LLM gen | MXFP8 (block-32) | ❓ Unknown — nsys profiling on B200 needed |
| B200 (SM100) | cuDNN (not used by vLLM) | MXFP8 (block-32) | ✅ Fused in GroupedGemmSwigluSm100 |

---

## Target Architecture: Qwen3.5-397B-A17B

```
hidden_size:                    4096
num_layers:                     60 (45 GDN + 15 full attention)
num_attention_heads:            32
num_kv_heads:                   2 (16:1 GQA)
head_dim:                       256
num_experts:                    512
num_experts_per_tok:            10
moe_intermediate_size:          1024 (per expert)
shared_expert_intermediate_size: 1024
vocab_size:                     248320
```

Layer pattern: `[GDN, GDN, GDN, FullAttn]` × 15 = 60 layers.
Every layer uses MoE MLP regardless of attention type (GDN or full attention only affects the attention mechanism).

### Parameter breakdown
| Component | Total | Active per token |
|-----------|-------|-----------------|
| Attention (all layers) | 6.4B | 6.4B (always on) |
| MLP (all 512 experts) | 387.3B | 8.3B (10 experts + shared) |
| Embeddings | 2.0B | 2.0B |
| Router | 0.13B | 0.13B |
| **Total** | **~396B** | **~17B** |

Attention = 38% of active params. MLP = 49%. But MLP takes more wall-clock time due to memory-boundedness of small expert GEMMs.

### Why attention is not the target
16:1 GQA (32 Q heads, 2 KV heads) makes K/V projections tiny (`[M, 4096] × [4096, 512]`). FlashAttention loads KV once and broadcasts to 16 Q heads. Well-optimized, not worth fusing.

### Shared expert is a free win
Every MoE layer has a shared expert (dense MLP, `intermediate_size=1024`) that processes all tokens regardless of routing. Identical to the dense fusion we already have working. Zero new kernel work.

---

## GEMM Shapes (Qwen3.5-397B, continuous batching M=1024)

### Routed experts (~200-400 active per layer)
- gate_up: `[5-50, 4096] × [4096, 2048]` — tiny, memory-bound
- down:    `[5-50, 1024] × [1024, 4096]` — tiny, memory-bound

### Shared expert (dense, all tokens)
- gate_up: `[1024, 4096] × [4096, 2048]`
- down:    `[1024, 1024] × [1024, 4096]`

### Attention projections (for reference)
- Q:  `[1024, 4096] × [4096, 8192]`
- K:  `[1024, 4096] × [4096, 512]` — tiny (2 KV heads)
- V:  `[1024, 4096] × [4096, 512]` — tiny
- O:  `[1024, 8192] × [8192, 4096]`

---

## vLLM MoE Internals

### Kernel stack
```
Qwen3NextSparseMoeBlock.forward()
  → SharedFusedMoE.forward()
    → shared expert: dense MLP (gate_up → SiLU*mul → down)
    → routed experts: FusedMoE.forward_native()
      → DefaultMoERunner.forward()
        → torch.ops.vllm.moe_forward_shared
          → moe_align_block_size()       # token sorting + padding
          → dispatch_fused_moe_kernel()   # Triton or FlashInfer CUTLASS
```

Six layers of abstraction. All in `vllm/model_executor/layers/fused_moe/fused_moe.py`.

### How dynamic routing works inside CUDA graphs
vLLM uses piecewise CUDA graphs: MoE layers are captured, attention runs eager.

`moe_align_block_size` (runs inside the graph) produces:
- **`sorted_token_ids`**: token indices sorted by expert, each expert's count padded to `BLOCK_SIZE_M`. Statically allocated at worst case: `topk_ids.numel() + num_experts * (block_size - 1)` (~42K slots for Qwen3.5).
- **`expert_ids`**: one expert ID per M-block. `-1` = skip (padding or expert not on this rank).
- **`num_tokens_post_padded`**: device scalar. Kernel early-exits beyond this.

Shapes never change — only contents. CUDA graph replays the same kernel with different device-side data each step.

Per-expert padding: each expert rounded up to `BLOCK_SIZE_M` (64 or 128). With 5-50 tokens per expert, 20-90% of compute per expert is on padding. Masked out in the kernel, not incorrect but wasteful.

### What the Triton fused_moe_kernel does per CTA
1. Read `expert_ids[pid_m]` → which expert's weights to load
2. If `-1`: write zeros, return
3. If `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded`: return (early exit)
4. Load token data via `sorted_token_ids` indirection into stacked activation tensor
5. Load expert weights via `expert_ids` offset into stacked weight tensor `[E, N, K]`
6. GEMM tile with masking for padding tokens
7. Optionally multiply by router weight, scatter-write output

---

## Fusion Strategy

### Injection point
Replace at `Qwen3NextSparseMoeBlock.forward()` level (same approach as our dense Qwen2 MLP injection), or drop in as a replacement for `dispatch_fused_moe_kernel` to reuse vLLM's existing `moe_align_block_size` preprocessing.

### What to fuse
**GEMM1 epilogue (gate_up):** grouped GEMM + SiLU*mul + block-FP8 quant → single CUTLASS kernel with custom epilogue.

### What stays the same
- Router (tiny linear, not worth fusing)
- Token sorting (`moe_align_block_size` — reuse vLLM's existing kernel)
- Down_proj grouped GEMM (input dequant is mainloop, not epilogue — not fusable)
- Unpermute + weighted sum
- Shared expert (existing dense fusion)
- FlashAttention

### Integration approach

**Option A (preferred): Use vLLM's sorted_token_ids interface**
- Reuse `moe_align_block_size` output directly
- CUTLASS kernel reads `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`
- Same scatter/gather pattern as the Triton kernel
- Same CUDA graph compatibility (static shapes, device-side metadata, early-exit)
- Minimal integration — swap the GEMM kernel, keep everything else

**Option B: Use CUTLASS GemmGrouped with problem list**
- Convert sorted_token_ids to CUTLASS problem descriptions (per-expert M, shared N/K)
- More CUTLASS-native but harder to integrate
- Dynamic problem count may break CUDA graph capture

### CUDA graph constraints (hard requirements)
- Static tensor shapes (pre-allocated at worst case)
- Device-side metadata only
- No host-device sync
- Early-exit for unused blocks
- Fixed grid size

### Open questions
1. Can a CUTLASS epilogue be attached to a kernel using vLLM's sorted_token_ids dispatch pattern (not standard GemmGrouped)?
2. Per-expert padding to BLOCK_SIZE_M is already baseline cost. Does CUTLASS require larger tiles?
3. Performance vs FlashInfer CUTLASS MoE (the actual default on H100) — need benchmarks.

---

## Platform Story

**H100 (SM90) — primary target, gap confirmed:**
FlashInfer CUTLASS is the default MoE backend. SwiGLU + block-FP8 quant between GEMM1 and GEMM2 are confirmed separate kernels (from source code analysis). We fuse them into GEMM1's epilogue. DeepSeek-style block-128 FP8 format. Cheap repack kernel needed between GEMM1 output (block-FP8) and GEMM2 input (row-FP8).

**B200 (SM100) — secondary target, gap uncertain:**
vLLM uses FlashInfer's TRT-LLM gen backend for FP8/FP4 MoE. The C++ internals are opaque — it may already fuse SwiGLU + quant via cuDNN's GroupedGemmSwiglu, or it may not. **An nsys trace of Qwen3.5-MoE inference on B200 would definitively reveal whether the SwiGLU and FP8 quant are separate kernel launches or fused into the GEMM1 epilogue.** If separate, our kernel fills the same gap in MXFP8 format. If already fused, SM100 is not a target.

**MoE multiplier:** Savings compound across ~200-400 active experts per layer, 60 layers. Even small per-kernel savings add up to meaningful e2e improvement.

---

## Rough Speedup Estimates

> These estimates are extrapolated from a single fused GEMM on dense Qwen3-8B — every number in this project was wrong until we measured it on the GPU, and these will be too.

| Scenario | Requant tax | Estimate | Notes |
|---|---|---|---|
| H100 dense (all 4 projections) | yes, per GEMM | 3-6% e2e | Measured 1-2% on 1 GEMM, ×4, minus requant overhead compounding |
| H100 MoE (all 4 projections) | yes, per GEMM | 3-6% e2e | Similar tensor sizes after TP, same pattern |
| B200 dense (all 4 projections) | no | 4-8% e2e | No requant, cleaner pipeline |
| B200 MoE (all 4 projections) | no | 4-8% e2e | Same as dense — the silu+quant tensor is similar size |

Note: MoE per-GPU activations are significantly smaller than dense (5MB vs 48MB for the gate_up intermediate), which may reduce the HBM savings. The actual impact depends on how much of the baseline cost is HBM traffic vs kernel overhead vs compute — only profiling can answer this.

---

## Next Steps

The cheapest way to validate the opportunity: monkey-patch vLLM's forward to skip every fusable kernel that sits between GEMMs (silu+quant, RMSNorm+quant, residual add+norm+quant) and benchmark the throughput difference. This gives the practical theoretical ceiling for all possible epilogue fusions combined, in under an hour, with zero kernel development. The gap between "all post-GEMM kernels removed" and baseline = the maximum e2e improvement any amount of fusion work could achieve.

After that:
1. nsys profile Qwen3.5-MoE on H100 to get the actual kernel breakdown and per-layer timing
2. nsys profile on B200 (if accessible) to determine whether TRT-LLM gen already fuses SwiGLU+quant
3. Verify CUTLASS custom epilogue feasibility with vLLM's sorted_token_ids dispatch pattern
4. Prototype: attach existing SwiGLU+FP8 epilogue to MoE grouped GEMM
5. Benchmark against FlashInfer CUTLASS on H100 (the actual baseline)
6. E2E integration with CUDA graph compatibility
7. If SM100 gap exists: separate kernel implementation for B200
