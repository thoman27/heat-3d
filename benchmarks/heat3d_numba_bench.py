import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
from scipy.ndimage import laplace

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions & Block Sizes
N     = 512
steps = 12
dx = dy = dz = 0.01      # in meters
dt = 0.001               # in seconds
alpha = 1.17e-4          # Thermal diffusivity (m^2/s) - Copper
r = alpha * dt / (dx**2) # CFL Stability Criterion for 3D: r <= 1/6 (~0.1667)

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive Heat Equation
# ---------------------------------------------------------
def solve_heat_naive_python_0(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        # Explicit 3D Finite Difference via 3-level nested loops
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                         + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                         + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];

        # Swap buffers for next time step
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_curr[i, j, k] =  r * (T_next[i+1, j, k] - 2.0 * T_next[i, j, k] + T_next[i-1, j, k]
                                         + T_next[i, j+1, k]  - 2.0 * T_next[i, j, k] + T_next[i, j-1, k]
                                         + T_next[i, j, k+1]  - 2.0 * T_next[i, j, k] + T_next[i, j, k-1]) + T_next[i, j, k];


        # Enforce Fixed Boundary Conditions (0°C edges)
        # (Outer boundaries i=0, i=nx-1, j=0, etc. remain 0.0)

    return T_curr

# ---------------------------------------------------------
# Baseline 1: Naive Heat Equation in Numba
# ---------------------------------------------------------
@njit
def solve_heat_numba_1(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        # Explicit 3D Finite Difference via 3-level nested loops
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                         + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                         + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];

        # Swap buffers for next time step
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_curr[i, j, k] =  r * (T_next[i+1, j, k] - 2.0 * T_next[i, j, k] + T_next[i-1, j, k]
                                         + T_next[i, j+1, k]  - 2.0 * T_next[i, j, k] + T_next[i, j-1, k]
                                         + T_next[i, j, k+1]  - 2.0 * T_next[i, j, k] + T_next[i, j, k-1]) + T_next[i, j, k];


        # Enforce Fixed Boundary Conditions (0°C edges)
        # (Outer boundaries i=0, i=nx-1, j=0, etc. remain 0.0)

    return T_curr

# ---------------------------------------------------------
# Baseline 3: Copy Buffer
# ---------------------------------------------------------
@njit
def solve_heat_copy_buffer_3(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps):
        # Explicit 3D Finite Difference via 3-level nested loops
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    d2T_dx2 = T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                    d2T_dy2 = T_curr[i, j+1, k] - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                    d2T_dz2 = T_curr[i, j, k+1] - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]

                    T_next[i, j, k] = T_curr[i, j, k] + r * (d2T_dx2 + d2T_dy2 + d2T_dz2)

        # Enforce Fixed Boundary Conditions (0°C edges)
        # (Outer boundaries i=0, i=nx-1, j=0, etc. remain 0.0)

        # Swap buffers for next time step
        T_curr = T_next.copy()

    return T_curr


# ---------------------------------------------------------
# Baseline Reference SciPy Laplace Solver
# ---------------------------------------------------------
def solve_heat_scipy(T, steps, r):
    T_curr = T.copy()
    for _ in range(steps):
        d2T = laplace(T_curr, mode='constant', cval=0.0)
        T_curr += np.dot(r, d2T)
        
    return T_curr


# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=512):
    global N
    N = matrix_size
    Nx = Ny = Nz = N
    # Initial setup of grid
    T_init = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    # Place a hot central source
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 <= 16:
                    T_init[i, j, k] = 100.0

    # Compute ground truth reference solution for correctness verification
    T_expected = solve_heat_scipy(T_init, steps, r)
    total_flops = 2.0 * (N ** 3) #TODO CALCULATE TOTAL FLOPS

    def measure(fn, reps=2):
        # Correctness check against reference matrix T_expected
        T_result = fn(T_init, steps, r)
        is_correct = np.allclose(T_result, T_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            fn(T_init, steps, r)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9 #TODO CALCULATE FLOPS PER SECOND ON COMPUTER

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python
    t0 = time.perf_counter()
    T_result = solve_heat_naive_python_0(T_init, steps, r)
    py_elapsed = time.perf_counter() - t0
    py_gflops  = (2.0 * (N ** 3) / py_elapsed) / 1e9 # TODO CALCULATE
    py_correct = np.allclose(T_result, T_expected, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_niave", solve_heat_numba_1),
        ("2_copy_buffer", solve_heat_copy_buffer_3),
        ("10_sci_py_laplacian", solve_heat_scipy),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results

# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for matrix dimension N
    N_input = 256
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])
        except ValueError:
            N_input = 256
    if len(sys.argv) > 2:
        try:
            img_path = sys.argv[2]
        except ValueError:
            img_path = f"heat3d_benchmark_N{N_input}.png"
    else:
        img_path = f"heat3d_benchmark_N{N_input}.png"

    print(f"\nRunning Heat Equation on 3D Grid {N_input}x{N_input}x{N_input}...\n")
    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    #divider = "-" * len(header)
    divider = "| :---" * (len(header.split("|"))-2) + " |"

    #print(divider)
    print(header)
    print(divider)

    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed

    #print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba Heat 3D Benchmark Performance (N={N_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    # save the plot as a PNG file
    plt.savefig(img_path, dpi=300)