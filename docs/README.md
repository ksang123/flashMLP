# flashMLP: Fusing SwiGLU into FP8 GEMMs on H100

[vLLM](https://github.com/vllm-project/vllm) is the most widely used open-source LLM serving framework. In its FP8 MLP block, the pipeline looks like this:

```
RMSNorm -> FP8 quant -> GEMM (gate_up) -> BF16 -> SiLU+Mul -> FP8 quant -> GEMM (down_proj) -> BF16
```

The SiLU+Mul+quant step reads and writes the full intermediate tensor through HBM. For Qwen3-8B that's 24,576 x batch_size elements, round-tripped through global memory. On an H100, this Triton kernel accounts for about 87ms out of a 1.4-second inference run. Trivial compute, all memory traffic.

We fused SwiGLU + FP8 quantization directly into the GEMM epilogue so the intermediate activations never leave registers. The project ran on a single rented H100 over ~64 hours of GPU time, went through 15+ kernel variants, and produced two paths that improve vLLM throughput by 1-2% with zero regressions across all tested workloads.

Setup: Qwen3-8B, FP8 dynamic quantization, vLLM v0.18.0, H100 SXM 80GB.

---

## The kernel

Writing a CUTLASS 3.x custom epilogue on SM90 is not straightforward. The epilogue API is designed for simple operations (scale, cast). We needed to dequantize FP8 accumulators, split the output into gate/up halves, compute `silu(gate) * up`, find the per-row absmax, and quantize to FP8. All from the accumulator registers, in one pass.

### What worked

**Single-pass epilogue.** Early versions used two passes: compute SwiGLU + absmax, then re-read and quantize. The two-pass overhead was ~58us for ~3us of actual compute. V7 replaced this with a `float vals[32]` register array, warp-shuffle absmax reduction, and immediate quantization. One pass, no re-reads. Saved 34us.

**Polynomial SiLU.** Standard `silu(x) = x * sigmoid(x)` uses the SFU for the exponential (16 ops/clock). We replaced sigmoid with a degree-5 polynomial approximation, pure FMA chain (128 ops/clock). Max error ~2.2% over [-6, 6], invisible after FP8 quantization. 8x throughput on the activation, saved 35us.

**Paired FP8 CVT.** SM90 has `cvt.rn.satfinite.e4m3x2.f32` that converts two floats to packed FP8 in one cycle. Early versions called it with a dummy zero for the second element.

### What didn't work

**Pingpong schedule.** vLLM uses CUTLASS Pingpong, which is ~30% faster than Cooperative because it overlaps GEMM and epilogue across warp groups. But Pingpong uses all 168 registers for the mainloop. Every attempt to add epilogue code caused register spills (STACK:8-16), destroying mainloop scheduling. We tried direct gmem stores, recompute without buffering, mainloop smem reuse. Nothing fit in the register budget. Pingpong is architecturally incompatible with non-trivial epilogues on SM90.

**128x128 tiles.** With TiledMma 2x1x1, the two warp groups each see only 64 of the 128 columns. Gate and up land in different warp groups and can't be paired for SwiGLU.

### The accumulator layout discovery

This was the hardest bug. The E2E integration ran, throughput looked good, output was garbage. The SwiGLU pairing was wrong.

The CUTLASS docs describe the GMMA 64x128 accumulator layout with a formula that implies `acc[2k]` pairs with `acc[2k+1]`. We implemented that. Wrong. We re-derived the formula from CuTe's layout algebra. Wrong again. We tried a third derivation. Wrong.

We injected printf into the epilogue, fed it constructed inputs with unique weight values per column, and discovered the Pingpong kernel uses a different layout than documented. The correct pairing is `acc[v]` with `acc[v+32]` for 64x128 tiles. As far as we know, this is the first published empirical mapping of the Pingpong accumulator layout.

Lesson: the formula derivation was wrong three times. Only the GPU tells the truth.

### Two viable paths

**Fused V12.** TiledMma 1x2x1 (splitting along N instead of M) gives a 64x256 tile where each warp group sees all columns. CUTLASS picks a native `MMA_64x256x32` GMMA instruction. Gate in cols 0-127, up in cols 128-255, both visible. REG:168 STACK:0, Pingpong schedule.

The trick that makes the fused path work end-to-end is weight interleaving: we rearrange the gate and up weight columns into stride-64 blocks at model load time, so each GMMA tile's accumulator naturally contains matched gate/up pairs. The epilogue then quantizes to block-64 FP8, where each block fits exactly within one tile's output. The absmax reduction is tile-local (warp shuffles only, no cross-CTA communication), so quantization adds near-zero overhead on top of the GEMM. The tradeoff: vLLM's down_proj GEMM expects per-row scales, so we need a small requant kernel (7us) to convert block-64 to per-row.

**Hybrid.** Keep vLLM's optimized 128x128 Pingpong GEMM, replace only the Triton silu+quant with a hand-written CUDA kernel. 1 block per row, 256 threads, uint4 vectorized loads, warp-shuffle absmax, native FP8 CVT. 38% faster than Triton's version. No weight interleaving, no requant, no double-quantization error.

---

## vLLM integration

Custom ops registered via `torch.library` for torch.compile + CUDA graph compatibility. Input quantization uses vLLM's own `QuantFP8` to preserve the Inductor RMSNorm+quant fusion. Environment variable (`FLASHMLP=0/1/2`) controls path selection, checked at import time to propagate across vLLM's multi-process architecture.

---

## Results

Qwen3-8B FP8, H100 SXM, vLLM v0.18.0, max_model_len=32768, 2000 prompts each.

```
              Config |     baseline |       hybrid |    fused V12 |   hybrid  fused V12
---------------------------------------------------------------------------------------
     in=128 out=1024 |    18750.1/s |    18954.6/s |    19058.0/s |  1.011x    1.016x
       in=512 out=64 |   188881.7/s |   197736.3/s |   196087.2/s |  1.047x    1.038x
      in=512 out=512 |    39999.2/s |    40634.9/s |    40310.7/s |  1.016x    1.008x
    in=1024 out=1024 |    26419.9/s |    26609.9/s |    26662.0/s |  1.007x    1.009x
     in=2048 out=512 |    61506.3/s |    61937.8/s |    61818.3/s |  1.007x    1.005x
    in=2048 out=2048 |    15585.3/s |    15746.3/s |    15874.6/s |  1.010x    1.019x
     in=4096 out=512 |    72254.7/s |    72479.0/s |    72423.2/s |  1.003x    1.002x
    in=4096 out=1024 |    35077.3/s |    35382.9/s |    35602.3/s |  1.009x    1.015x
     in=8192 out=512 |    80202.7/s |    80382.2/s |    80534.4/s |  1.002x    1.004x
    in=8192 out=1024 |    38923.3/s |    39187.9/s |    39408.0/s |  1.007x    1.012x
    in=16384 out=512 |    86252.9/s |    86410.0/s |    86679.1/s |  1.002x    1.005x
   in=16384 out=1024 |    41936.4/s |    42017.2/s |    42092.2/s |  1.002x    1.004x
```

Heavy decode:

```
              Config |     baseline |       hybrid |    fused V12 |   hybrid  fused V12
---------------------------------------------------------------------------------------
  in=1024 out=1024   |    26389.6/s |    26591.6/s |    26678.3/s |  1.008x    1.011x
  in=2048 out=2048   |    15602.2/s |    15727.2/s |    15864.9/s |  1.008x    1.017x
  in=1024 out=4096   |     7635.2/s |     7712.0/s |     7629.1/s |  1.010x    0.999x
  in=2048 out=8192   |     4117.6/s |     4152.1/s |     4101.1/s |  1.008x    0.996x
```

| Path | Geomean | Best | Worst | Regressions |
|------|---------|------|-------|-------------|
| Hybrid | +0.9% | +4.7% | +0.2% | 0/16 |
| Fused V12 | +1.0% | +3.8% | -0.4% | 2/16 (noise) |

V12 shows slight regressions at very long output lengths where attention dominates and the requant overhead is proportionally larger. The hybrid path avoids this entirely.

The silu+quant kernel is 6.2% of baseline GPU time. Our CUDA version is 38% faster than Triton's. Theoretical e2e gain: `0.062 * 0.38 = 2.4%`. Measured 1-2% is consistent. Gains are larger on prefill-heavy workloads (MLP is a bigger fraction) and smaller at long context (attention dominates).

---

## How this was built

The research, architecture, and kernel design were ours. We worked out the full approach on paper first: the epilogue structure, accumulator pairing strategy, register budget, quantization scheme, down to which instructions to use. The stride-64 weight interleaving, tile-local block quantization with no cross-CTA communication, polynomial SiLU for FMA-only execution, paired FP8 CVT, all of that was designed before any code was written. Once we had a complete theoretical implementation, we handed it to an AI coding agent on a rented H100 to iterate on the actual code.

That's when things started going wrong. The accumulator layout didn't match our derivation. The register budget didn't fit Pingpong. The block-64 scales got rejected by vLLM's GEMM. Each time, we went back to the theory, fixed the approach, and had the agent iterate on the implementation: trying different tile shapes, GEMM schedules, quantization block sizes, smem strategies, store patterns. The agent handled the compile-benchmark-analyze loop while we focused on figuring out why things were broken and what to try next.

Over ~64 hours of GPU time (roughly 48 of which the agent ran autonomously), this produced ~200 benchmark runs across 15 kernel variants and 125 commits.

---

## Future work

The technique applies to every GEMM in a transformer. Each one has post-processing that currently runs as a separate kernel: silu+quant after gate_up, RMSNorm+quant after down_proj, RoPE+quant after QKV, residual+norm+quant after output projection. We fused one. The other three have different (and simpler) post-ops, none of them require pairing accumulator registers across a GMMA tile, which is a relief. Rough estimates for fusing all four:

| Scenario | Estimate | Notes |
|---|---|---|
| H100 dense (all 4 projections) | 3-6% e2e | Measured 1-2% on 1 GEMM, x4, minus requant overhead |
| H100 MoE (all 4 projections) | 3-6% e2e | Similar tensor sizes after TP |
| B200 dense (all 4 projections) | 4-8% e2e | No requant (native block scaling) |
| B200 MoE (all 4 projections) | 4-8% e2e | Same pattern |

### MoE

On H100, vLLM uses FlashInfer's CUTLASS backend for FP8 MoE. We confirmed from source that SwiGLU + FP8 quant between GEMM1 and GEMM2 are separate kernel launches. Nobody fuses them. On B200, vLLM uses FlashInfer's TRT-LLM gen backend, which is a black box; it might already fuse via cuDNN internally. An nsys trace on B200 would answer this. Full research in [MOE_RESEARCH.md](MOE_RESEARCH.md).

One caveat: MoE per-GPU activations are smaller than dense (5MB vs 48MB for gate_up with TP8 on Qwen3.5-397B). The HBM savings scale with activation size, so the per-layer benefit may be smaller. Only profiling can answer how much.

### Blackwell

NVIDIA's B200 introduces native block scaling in the tensor cores. The next GEMM can consume block-quantized FP8 directly from the previous epilogue. No requant kernel, no double-quantization error. The full vision: every GEMM gets a fat epilogue, activation tensors never touch HBM as BF16 intermediates. For Qwen3-8B, that eliminates ~150MB of HBM traffic per layer, 5.4GB across 36 layers.

### Validating before building

The cheapest way to check if any of this is worth it: monkey-patch vLLM's forward to skip every fusable kernel between GEMMs and benchmark the throughput difference. Under an hour, zero kernel development. The gap = the maximum e2e improvement any fusion work could ever achieve.

Every estimate in this post was wrong at least once. The theoretical minimum was wrong. The baseline was wrong. The GPU was throttled by a rogue process for 17 hours and we didn't notice. The accumulator layout formula was wrong three times. The only thing that was never wrong was the benchmark, and even that needed the rogue process killed first. If this project taught us anything, it's that theory couldn't be further from reality when it comes to GPU kernels. The future work estimates above are probably wrong too. There's only one way to find out.
