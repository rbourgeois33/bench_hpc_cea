# matvec_bw — memory-bandwidth benchmark of y = A·x on GPU (Kokkos)

A is N×M (default N = 10 000 000, M = 32, compile-time), float32.

| kernel | what it does |
|---|---|
| `naive` | `y(i) += A(i,j)*x(j)` for every j → 32 read-modify-writes of `y(i)` in global memory |
| `less_redundant` | accumulate in a register, write `y(i)` once |
| `shared` | `TeamPolicy`: `x` is loaded **once per team (= CUDA block)** into team scratch (= block shared memory), then each thread of the block computes its rows against that copy |

Both `LayoutLeft` and `LayoutRight` are benchmarked in every run.

## Build (RTX 6000 Ada, out of the box)

Requirements (Kokkos 5.2.2): CMake ≥ 3.25.2, CUDA ≥ 12.2, GCC ≥ 10.4 as host compiler.

```bash
cmake -B build            # CUDA + Kokkos_ARCH_ADA89, Release, Kokkos 5.2.2 fetched
cmake --build build -j
./build/matvec_bench
```

No need to set `CMAKE_CXX_COMPILER` to `nvcc_wrapper`: Kokkos installs its compiler launcher globally.

Options: `-DMATVEC_BACKEND=SERIAL|OPENMP` for a CPU smoke test, `-DMATVEC_KERNEL_REPORT=ON` to compile with
`-Xptxas -v` (stack frame / spill / register usage per kernel), `-DFETCHCONTENT_SOURCE_DIR_KOKKOS=/path` to use a local Kokkos.

## Run

```
./build/matvec_bench [--N 10000000] [--warmup 5] [--min-reps 30] [--min-time 3.0]
                     [--peak-bw 960] [--rows-per-team 0]
```

* Each kernel runs `warmup` untimed products, then repeats until **both** `min-reps` and `min-time` are reached.
  Each product is fenced. Median, min and CV are reported; if CV is high, raise `--min-time`.
* `y` is zeroed (untimed) before **every** product of **every** kernel, so all kernels start from the same memory state.
* Bandwidth = `((N*M + M) + N) * 4 B / median_time`, in GB/s (1e9), for every kernel (including naive).
  % is relative to `--peak-bw` (960 GB/s, RTX 6000 Ada spec).
* Every kernel's output is checked against a reference (`relerr` column; a warning is printed above 1e-5).
* `--N` makes the problem bigger. Memory needed ≈ `N*M*4 + 2*N*4` bytes (one layout allocated at a time), i.e. about 1.36 GB at the default N; 48 GB allows N up to roughly 3×10⁸.

### Shared kernel: team size
The team size is `Kokkos::AUTO`. At startup the program prints:

```
shared config: team_size AUTO -> T (team_size_max X), rows_per_team R, league_size L, team scratch B
```

`T` comes from `team_size_recommended(functor, ParallelForTag)` with the same functor and scratch size.
In Kokkos 5.2.2 this is exactly the call `parallel_for` makes to resolve `AUTO`, so it is the value actually used.
Rows are strided over the team threads (`TeamThreadRange`), so results are correct for any `rows_per_team`.
By default `rows_per_team = T` (one row per thread).

The load uses `single(PerTeam)` followed by `team_barrier()`. `single` only synchronises its own warp, so with
team sizes above 32 threads in other warps could otherwise read the shared copy before it is written.

## Plot

At the end, the benchmark prints a ready-to-paste command:

```bash
python3 scripts/plot_bw.py --peak 960 "naive (Left)=123.4" "shared (Right)=456.7" --out bw.png
```

Arguments are `LABEL=VALUE` (GB/s). `--peak` draws the GPU peak line and labels each bar with its %. Needs matplotlib.# bench_hpc_cea
