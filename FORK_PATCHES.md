# Swift fork integration for MLX v0.32.4

Upstream Swift cutoff: `19601207e9a0de51e03ee6ec0c3c5f3784275075`
(release 0.32.3). The core submodule incorporates MLX v0.32.4 and the retained
patch inventory in its `FORK_PATCHES.md`. The mlx-c submodule is the matching
upstream binding refresh `57731b00e8bed710e0750e7356da9f18f8eb3e20`.
Both fork and upstream histories are retained for review.

| Previous fork commit | Disposition |
| --- | --- |
| fdb1a7c | Retain baseline Swift 6.0 manifest and the Swift 6.3 CUDA manifest. |
| a1ea6fd | Retain Linux fast-symbol linkage and the CMake bridge. |
| c2dbb6f | Replace temporary runtime API adaptations with matching upstream v0.32.4 C bindings. |
| d4f707f | Replace stream allocation with upstream executor-safe thread-unsafe streams and pooling. |
| 802bb53 | Retain actor isolation on asynchronous stream scopes. |
| 08e9ba7 | Retain opt-in quantized custom-kernel headers in core. |
| d353dcc | Retain distinct task-local CPU/GPU streams through upstream stream pairs. |
| 0ae3580 | Replace eager availability checks with upstream lazy companion streams. |
| f92684f | Retain source-reproducible NVFP4 staging in core. |
| 2878bab | Regenerate AOT sources and public/framework headers from the new submodules. |
| b28ba70 | Retain surviving fast-kernel tests and formatting. |
| f8536fe | Retain core QMV dispatch correctness and execution regressions. |
| 3e6df6d | Retain core Linux matmul compatibility. |
| 5bf3e46 | Retain source-level H3/M4 and NAX optimizations, adapting to upstream tiling. |
| 30be5d5 | Retain core non-macOS Apple CPU-JIT availability guard. |
| abdaea7 | Preserve cross-executor compile-cache erasure through namespaced private bridge functions alongside the standard mlx-c API. |
| 0b491c7 | Preserve concurrent stateless compilation with core thread-local caches. Stateful calls adopt upstream recursive instance locks and outer eval-lock ordering. |
| 7558b9c | Retain concurrent-first-trace, shape, numerical, cross-thread-erasure tests and the opt-in overhead benchmark. |
| 001d1aa | Preserve the public reusable `Stream.Context` and `withDefaultStream` APIs over upstream stream pairs/pooling. |

Source generation includes the new always-required gather, attention-VJP, and
gated-delta kernels. Fast-fence kernels are conditional on Metal 3.2 so the
baseline macOS 14 shader build remains valid. Framework and SwiftPM manifests
both report the actual MLX 0.32.4 core version.

The FP8 byte-decoding fixture follows v0.32.4's corrected E4M3FN semantics:
0x7f and 0xff decode to NaN, and 0x7e/0xfe remain finite. The generation-17
NAX regression is included in the Xcode test target, alongside the owned
stream/context and compile-cache regressions.

Both Swift manifests exclude the additional CUDA host sources introduced by
v0.32.4 on Apple platforms. The baseline manifest uses the matching upstream
C distributed bindings and selects the conditional CPU-JIT wrapper once.
Both manifest paths build on the local Swift 6.4 toolchain; that does not
replace execution with an actual Swift 6.0 or Linux/CUDA toolchain.

The Linux CLI integration additionally excludes the v0.32.4 Metal
`cross_entropy.cpp` and `gated_delta_update.cpp` host implementations from
CPU/CUDA SwiftPM builds. The evaluated Apple package is byte-identical before
and after this platform-only manifest correction.

The core pin also includes the explicit dependent-type `typename` required
by the downstream Linux CPU compiler. This changes no GPU source or shader.

The generation-17 NAX guard now applies only to quantized dispatch in core.
Floating-point matmul, attention, and gated delta use upstream availability.
The Swift regression checks both FP16 and BF16 floating-point matmul and
1/2/4/8-bit quantized matmul at the historical wide-output threshold.
The Xcode fixture plan selects FP32 precision, matching the official core
runner; production inference defaults remain unchanged.

The mlx-c pin advances from the matching upstream cutoff above to the owned
portable-I/O patch. Its five error-message formatting calls use string
construction, preserving their diagnostics on Ubuntu 22.04's libstdc++ without
requiring `std::format`. The C-binding fork records its cutoff and regression
command in `FORK_PATCHES.md`. The compile-cache bridge also drops an unused
private umbrella include; it uses the public compile declaration and core
compile-cache API directly. These host-only changes alter no generated Metal
sources or inference arithmetic.
