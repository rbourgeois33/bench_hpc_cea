// Matrix-vector product y = A x, A is (N x M), x is (M), y is (N), float32.
//
// Kernels
//   naive          : y(i) += A(i,j)*x(j) for each j   -> M read-modify-writes of y(i) in global memory
//   less_redundant : accumulate in a register, write y(i) once
//   shared         : TeamPolicy; x is loaded once per team (= CUDA block) into
//                    team scratch (= block shared memory), then each thread
//                    computes its rows against the shared copy
//
// Bandwidth model (identical for ALL kernels, including naive):
//   bytes per product = ((N*M) + M) floats read + N floats written
//   BW = bytes / time_per_product                          [GB/s, 1 GB = 1e9 B]
//
// N and M are runtime inputs.
// Both LayoutLeft and LayoutRight are benchmarked in one run.

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using IndexT    = std::int64_t;

using RangeP    = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<IndexT>>;
using TeamP     = Kokkos::TeamPolicy<ExecSpace>;
using Member    = TeamP::member_type;
using ScratchV  = Kokkos::View<float*, ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

constexpr std::size_t FS = sizeof(float);

// ---------------------------------------------------------------------------
struct Options {
  IndexT N             = 10'000'000;
  int M                = 32;
  int warmup           = 5;
  int min_reps         = 100;
  double min_time_s    = 3.0;    // keep repeating until both min_reps and min_time are reached
  double peak_bw_gbs   = 960.0;  // RTX 6000 Ada spec memory bandwidth
  IndexT rows_per_team = 0;      // shared kernel: 0 -> equal to the AUTO team size (1 row per thread)
  // Memory hierarchy constants (defaults: Ada Lovelace / RTX 6000 Ada).
  int sector_bytes     = 32;     // L1/L2 sector
  int cache_line_bytes = 128;    // L1/L2 cache line = 4 sectors
  int l1_kb            = 128;    // per SM, unified with shared memory (data part = 128 KB - carveout)
};

void usage(const char* prog) {
  std::printf(
      "Usage: %s [options] [kokkos options]\n"
      "  --N <int>                rows (default 10000000)\n"
      "  --M <int>                columns (default 32)\n"
      "  --warmup <int>           untimed warm-up products per kernel (default 5)\n"
      "  --min-reps <int>         minimum timed products per kernel (default 30)\n"
      "  --min-time <sec>         minimum total timed seconds per kernel (default 3.0)\n"
      "  --peak-bw <GB/s>         GPU peak memory bandwidth (default 960, RTX 6000 Ada)\n"
      "  --rows-per-team <int>    shared kernel rows per team (default 0 = AUTO team size)\n"
      "  --sector-bytes <int>     cache sector size (default 32, Ada)\n"
      "  --cache-line-bytes <int> cache line size (default 128, Ada)\n"
      "  --l1-kb <int>            L1 size per SM in KB (default 128, Ada)\n",
      prog);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int k = 1; k < argc; ++k) {
    std::string a = argv[k];
    auto next = [&]() -> std::string {
      if (k + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
      return argv[++k];
    };
    if      (a == "--N")                o.N                = std::stoll(next());
    else if (a == "--M")                o.M                = std::stoi(next());
    else if (a == "--warmup")           o.warmup           = std::stoi(next());
    else if (a == "--min-reps")         o.min_reps         = std::stoi(next());
    else if (a == "--min-time")         o.min_time_s       = std::stod(next());
    else if (a == "--peak-bw")          o.peak_bw_gbs      = std::stod(next());
    else if (a == "--rows-per-team")    o.rows_per_team    = std::stoll(next());
    else if (a == "--sector-bytes")     o.sector_bytes     = std::stoi(next());
    else if (a == "--cache-line-bytes") o.cache_line_bytes = std::stoi(next());
    else if (a == "--l1-kb")            o.l1_kb            = std::stoi(next());
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); std::exit(1); }
  }
  if (o.N <= 0 || o.min_reps <= 0 || o.warmup < 0 || o.rows_per_team < 0 || o.M <= 0 || o.sector_bytes <= 0 ||
      o.cache_line_bytes <= 0 || o.l1_kb <= 0) {
    std::fprintf(stderr, "invalid option value\n");
    std::exit(1);
  }
  return o;
}

// ---------------------------------------------------------------------------
// Kernels (M is a runtime value, taken from A.extent(1))
// ---------------------------------------------------------------------------

for (int i = 1; i < N; ++i) {
    for (int j = 1; j < M; ++j) {            
       y[i] += A[j+M*i] * x[j]
    }
}

template <class Mat, class Vec>
void kernel_naive(const Mat& A, const Vec& x, const Vec& y) {
  const int M = A.extent_int(1);
  const int N = A.extent_int(0);

  auto policy = Kokkos::RangePolicy(0, N);

  Kokkos::parallel_for("naive", 
    policy, 
    KOKKOS_LAMBDA(const int i) 
    {
        for (int j = 0; j < M; ++j) {
            y(i) += A(i, j) * x(j);
        }
  });

}

template <class Mat, class Vec>
void kernel_less_redundant(const Mat& A, const Vec& x, const Vec& y) {
  const int M = A.extent_int(1);
  const int N = A.extent_int(0);

  auto policy = Kokkos::RangePolicy(0, N);
  Kokkos::parallel_for("less_redundant", 
    policy, 
    KOKKOS_LAMBDA(const int i) 
    {
        float s = 0.0f;
        for (int j = 0; j < M; ++j){
             s += A(i, j) * x(j);
        }
        y(i) = s;
    });
}
/* 
template <class Mat>
struct SharedKernel {
  Mat A;
  Vec x;
  Vec y;
  IndexT N;
  int M;
  IndexT rows_per_team;

  KOKKOS_INLINE_FUNCTION
  void operator()(const Member& t) const {
    const IndexT begin = static_cast<IndexT>(t.league_rank()) * rows_per_team;
    const IndexT end   = Kokkos::min(begin + rows_per_team, N);

    // Same address for every thread of the block: block shared memory.
    ScratchV xs(t.team_shmem(), M);

    // Load x once per team (thread 0 only)...
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      for (int j = 0; j < M; ++j) xs(j) = x(j);
    });
    // ...and make every thread of the block wait for it. single(PerTeam) only
    // syncs its own warp; with team_size > 32 the other warps would race.
    t.team_barrier();

    // Rows [begin, end) are strided across the threads of the team.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, begin, end), [&](const IndexT i) {
      float s = 0.0f;
      for (int j = 0; j < M; ++j) s += A(i, j) * xs(j);
      y(i) = s;
    });
  }
}; */

// ---------------------------------------------------------------------------
// Memory geometry report
// ---------------------------------------------------------------------------
struct Geometry {
  double stride_threads_B = 0;  // A(i,j) -> A(i+1,j)
  double stride_inner_B   = 0;  // A(i,j) -> A(i,j+1)
};

Geometry geometry(const Options& o, bool right) {
  Geometry g;
  g.stride_threads_B = right ? static_cast<double>(o.M) * FS : FS;
  g.stride_inner_B   = right ? FS : static_cast<double>(o.N) * FS;
  return g;
}

std::string fmt_bytes(double b) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(b < 1024 ? 0 : 2);
  if (b < 1024) s << b << " B";
  else if (b < 1024.0 * 1024) s << b / 1024 << " KB";
  else if (b < 1024.0 * 1024 * 1024) s << b / (1024.0 * 1024) << " MB";
  else s << b / (1024.0 * 1024 * 1024) << " GB";
  return s.str();
}

void print_geometry(const Options& o, bool right, int warp) {
  const Geometry g   = geometry(o, right);
  const double line  = o.cache_line_bytes;
  const double sect  = o.sector_bytes;
  const double l1    = o.l1_kb * 1024.0;
  const double row_B = static_cast<double>(o.M) * FS;

  auto line_info = [&](double stride) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(3) << stride / sect << " sectors, " << stride / line
      << " cache lines";
    // cache lines touched by one warp for one load instruction
    const double span  = warp * stride;
    const double lines = std::min<double>(warp, std::ceil(span / line));
    s << std::setprecision(0) << "; one warp (" << warp << " thr) spans " << fmt_bytes(span) << " -> "
      << lines << " cache line" << (lines > 1 ? "s" : "");
    const double thr_per_line = line / stride;
    if (thr_per_line >= warp)    s << std::setprecision(1) << " (" << thr_per_line << " thr/line, fully coalesced)";
    else if (thr_per_line > 1.0) s << std::setprecision(1) << " (" << thr_per_line << " thr/line, partially coalesced)";
    else                         s << " (every thread a distinct line, NOT coalesced)";
    return s.str();
  };

  std::cout << std::left
            //<< "  memory geometry (sector " << o.sector_bytes << " B, cache line " << o.cache_line_bytes
            //<< " B, L1 " << o.l1_kb << " KB/SM, warp " << warp << "):\n"
            << "    A  stride between threads (i->i+1) : " << std::setw(8) << fmt_bytes(g.stride_threads_B)
            << " = " << line_info(g.stride_threads_B) << "\n"
            << "    A  stride within thread  (j->j+1) : " << std::setw(8) << fmt_bytes(g.stride_inner_B)
            << " = " << std::fixed << std::setprecision(3) << g.stride_inner_B / line << " cache lines"
            << (g.stride_inner_B <= line ? " (same/adjacent line)" : " (new line every j)") << "\n";
            //<< "    y  stride between threads (i->i+1) : " << std::setw(8) << fmt_bytes(FS) << " = "
            //<< line_info(FS) << "\n"
            //<< "    x  stride between threads          : 0 B (all threads load the same "
            //<< fmt_bytes(row_B) << "; 'shared' loads it once per block)\n"
            //<< "    one row of A (M floats)            : " << fmt_bytes(row_B) << " = " << std::setprecision(3)
            //<< row_B / line << " cache lines; L1 holds " << std::setprecision(0) << l1 / line
            //<< " cache lines = " << std::floor(l1 / row_B) << " rows (before shared-memory carveout)\n";
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
struct Stats {
  double median_s = 0, min_s = 0, mean_s = 0, cv_pct = 0;
  int reps = 0;
};

Stats compute_stats(std::vector<double> t) {
  Stats s;
  s.reps = static_cast<int>(t.size());
  std::sort(t.begin(), t.end());
  const std::size_t n = t.size();
  s.median_s = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
  s.min_s    = t.front();
  s.mean_s   = std::accumulate(t.begin(), t.end(), 0.0) / n;
  double var = 0;
  for (double v : t) var += (v - s.mean_s) * (v - s.mean_s);
  var /= (n > 1 ? n - 1 : 1);
  s.cv_pct = 100.0 * std::sqrt(var) / s.mean_s;
  return s;
}

template <class F, class Vec>
Stats bench(const Options& o, const Vec& y, F&& run_once) {
  auto one = [&]() {
    // Reset y before EVERY product for EVERY kernel (untimed), so that all kernels
    // start from the same memory state (naive needs y == 0 anyway).
    Kokkos::deep_copy(y, 0.0f);
    Kokkos::fence();
    Kokkos::Timer timer;
    run_once();
    Kokkos::fence();
    return timer.seconds();
  };
  for (int w = 0; w < o.warmup; ++w) one();
  std::vector<double> times;
  double total = 0;
  while (static_cast<int>(times.size()) < o.min_reps || total < o.min_time_s) {
    double dt = one();
    times.push_back(dt);
    total += dt;
  }
  return compute_stats(std::move(times));
}
template <class Vec>
float max_rel_diff(const Vec& a, const Vec& b) {
  float r = 0.0f;
  Kokkos::parallel_reduce("check", RangeP(0, a.extent(0)), KOKKOS_LAMBDA(const IndexT i, float& m) {
    const float d = Kokkos::abs(a(i) - b(i)) / Kokkos::max(Kokkos::abs(b(i)), 1.0e-30f);
    if (d > m) m = d;
  }, Kokkos::Max<float>(r));
  return r;
}

struct Result {
  std::string layout, kernel;
  Stats st;
  double bw_gbs = 0, pct_peak = 0, stride_B = 0;
  float rel_err = 0;
};

// ---------------------------------------------------------------------------
template <class Layout, class MemoryTrait = Kokkos::MemoryTraits<0>>
void run_layout(const Options& o, const char* layout_name, bool right, std::vector<Result>& results) {
  using Mat = Kokkos::View<float**, Layout, MemSpace, MemoryTrait>;
  using Vec = Kokkos::View<float*, MemSpace, MemoryTrait>;

  const IndexT N = o.N;
  const int M    = o.M;

  std::cout << "\n================ Layout" << layout_name << " (N = " << N << ", M = " << M
            << ") ================\n";
#ifdef KOKKOS_ENABLE_CUDA
  print_geometry(o, right, TeamP::vector_length_max());  // CUDA warp size
#else
  print_geometry(o, right, 32);  // not a GPU backend: assume a 32-thread warp
#endif

  Mat A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), N, M);
  Vec x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), M);
  Vec y("y", N);
  Vec y_ref("y_ref", N);

  // Random fill on the device (first touch on the GPU).
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(12345);
  Kokkos::fill_random(A, pool, 0.0f, 1.0f);
  Kokkos::fill_random(x, pool, 0.0f, 1.0f);
  Kokkos::fence();

  // Reference result.
  kernel_less_redundant(A, x, y_ref);
  Kokkos::fence();

  // Bandwidth model: ((N*M) + M) reads + N writes, float32.
  const double bytes = static_cast<double>((N * M + M) + N) * FS;
  const double stride = geometry(o, right).stride_threads_B;

  auto record = [&](const char* name, const Stats& st) {
    Result r;
    r.layout   = layout_name;
    r.kernel   = name;
    r.st       = st;
    r.bw_gbs   = bytes / st.median_s / 1.0e9;
    r.pct_peak = 100.0 * r.bw_gbs / o.peak_bw_gbs;
    r.stride_B = stride;
    r.rel_err  = max_rel_diff(y, y_ref);
    std::cout << "  " << std::left << std::setw(16) << name << " done: " << st.reps
              << " reps, median " << std::fixed << std::setprecision(3) << st.median_s * 1e3
              << " ms, CV " << std::setprecision(2) << st.cv_pct << " %, max rel err vs ref "
              << std::scientific << std::setprecision(2) << r.rel_err << std::defaultfloat << "\n";
    results.push_back(r);
  };

  record("naive", bench(o, y, [&]() { kernel_naive(A, x, y); }));
  record("less_redundant", bench(o, y, [&]() { kernel_less_redundant(A, x, y); }));

//   {
//     const std::size_t scratch_bytes = ScratchV::shmem_size(M);
//     SharedKernel<Mat> f{A, x, y, N, M, 1};

//     // Kokkos resolves AUTO inside parallel_for by calling exactly
//     // team_size_recommended(functor, ParallelForTag) on the policy, which depends on
//     // the functor type and the team scratch size (not on league size). We query the
//     // same thing here so the printed value is the one used.
//     TeamP probe(1, Kokkos::AUTO);
//     probe.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
//     const int ts_auto = probe.team_size_recommended(f, Kokkos::ParallelForTag());
//     const int ts_max  = probe.team_size_max(f, Kokkos::ParallelForTag());

//     f.rows_per_team = (o.rows_per_team > 0) ? o.rows_per_team : static_cast<IndexT>(ts_auto);
//     const IndexT league = (N + f.rows_per_team - 1) / f.rows_per_team;

//     std::cout << "  shared config: team_size AUTO -> " << ts_auto << " (team_size_max " << ts_max
//               << "), rows_per_team " << f.rows_per_team
//               << (o.rows_per_team > 0 ? " (user)" : " (= AUTO team size)") << ", league_size " << league
//               << ", team scratch " << scratch_bytes << " B (max level-0 " << TeamP::scratch_size_max(0)
//               << " B)\n";
//     if (league > static_cast<IndexT>(std::numeric_limits<int>::max())) {
//       std::cerr << "league_size overflows int, increase --rows-per-team\n";
//       std::exit(1);
//     }

//     TeamP policy(static_cast<int>(league), Kokkos::AUTO);
//     policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
//     record("shared", bench(o, y, [&]() { Kokkos::parallel_for("shared", policy, f); }));
//   }
}

void print_table(const Options& o, const std::vector<Result>& res) {
  const IndexT M = o.M;
  const double gb = static_cast<double>((o.N * M + M) + o.N) * FS / 1.0e9;
  std::cout << "\n==================================== RESULTS ====================================\n"
            << "N = " << o.N << ", M = " << M << ", float32, volume per product = " << std::fixed
            << std::setprecision(3) << gb << " GB (((N*M)+M) reads + N writes)\n"
            << "Peak GPU memory bandwidth = " << std::setprecision(1) << o.peak_bw_gbs << " GB/s\n"
            << "Cache: sector " << o.sector_bytes << " B, cache line " << o.cache_line_bytes << " B, L1 "
            << o.l1_kb << " KB/SM. stride = A stride between consecutive threads\n\n";
  std::cout << std::left << std::setw(8) << "layout" << std::setw(16) << "kernel" << std::right
            << std::setw(10) << "stride[B]" << std::setw(8) << "reps" << std::setw(13) << "median[ms]"
            << std::setw(11) << "min[ms]" << std::setw(8) << "CV[%]" << std::setw(12) << "BW[GB/s]"
            << std::setw(10) << "%peak" << std::setw(10) << "relerr" << "\n"
            << std::string(106, '-') << "\n";
  for (const auto& r : res) {
    std::cout << std::left << std::setw(8) << r.layout << std::setw(16) << r.kernel << std::right
              << std::fixed << std::setprecision(0) << std::setw(10) << r.stride_B << std::setw(8)
              << r.st.reps << std::setprecision(3) << std::setw(13) << r.st.median_s * 1e3 << std::setw(11)
              << r.st.min_s * 1e3 << std::setprecision(2) << std::setw(8) << r.st.cv_pct
              << std::setprecision(1) << std::setw(12) << r.bw_gbs << std::setw(9) << r.pct_peak << "%"
              << std::scientific << std::setprecision(1) << std::setw(10) << r.rel_err << std::defaultfloat
              << "\n";
  }
  std::cout << "\nBW uses the MEDIAN time per product.\n";

  bool bad = false;
  for (const auto& r : res) bad |= (r.rel_err > 1.0e-5f);
  if (bad) std::cout << "WARNING: a kernel differs from the reference by > 1e-5 (relative). Check it!\n";

  std::cout << "\nPlot command:\n  python3 ../plot.py --peak " << std::setprecision(1) << std::fixed
            << o.peak_bw_gbs;
  for (const auto& r : res) std::cout << " \"" << r.kernel << " (" << r.layout << ")=" << r.bw_gbs << "\"";
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);  // consumes --kokkos-* args
  const Options o = parse(argc, argv);

  std::cout << "Kokkos " << KOKKOS_VERSION_MAJOR << "." << KOKKOS_VERSION_MINOR << "."
            << KOKKOS_VERSION_PATCH << ", execution space: " << ExecSpace::name() << "\n";
#ifndef KOKKOS_ENABLE_CUDA
  std::cout << "WARNING: CUDA backend not enabled, this is NOT a GPU run.\n";
#endif
  Kokkos::print_configuration(std::cout, false);

  const double gb_A    = static_cast<double>(o.N) * o.M * FS / 1e9;
  const double gb_peak = gb_A + 2.0 * static_cast<double>(o.N) * FS / 1e9;
  std::cout << "Matrix A: " << o.N << " x " << o.M << " = " << gb_A
            << " GB; one layout allocated at a time, ~" << gb_peak << " GB device memory peak\n";

  std::vector<Result> results;
  run_layout<Kokkos::LayoutLeft>(o, "Base", false, results);
  //run_layout<Kokkos::LayoutRight>(o, "Layout Right", true, results);
  run_layout<Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Restrict>>(o, "Restrict", false, results);
  //run_layout<Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Restrict>>(o, "Right", true, results);
  print_table(o, results);
  return 0;
}