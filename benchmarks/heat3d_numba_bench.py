import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange, stencil
import numpy as np
from scipy.ndimage import laplace

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions & Block Sizes
N     = 512
steps = 100 #TODO MAKE 20 AGAIN
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

    
    if steps % 2 == 1:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                        + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                        + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];
        return T_next

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

    if steps % 2 == 1:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                        + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                        + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];
        return T_next

    return T_curr

@njit
def solve_heat_numba_reduce_op_1(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        # Explicit 3D Finite Difference via 3-level nested loops
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] + T_curr[i-1, j, k]
                                         + T_curr[i, j+1, k]  + T_curr[i, j-1, k]
                                         + T_curr[i, j, k+1]  + T_curr[i, j, k-1] - 6.0 * T_curr[i, j, k]) + T_curr[i, j, k];

        # Swap buffers for next time step
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_curr[i, j, k] =  r * (T_next[i+1, j, k] + T_next[i-1, j, k]
                                         + T_next[i, j+1, k]  + T_next[i, j-1, k]
                                         + T_next[i, j, k+1]  + T_next[i, j, k-1] - 6.0 * T_next[i, j, k]) + T_next[i, j, k];

        if steps % 2 == 1:
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        T_next[i, j, k] =  r * (T_curr[i+1, j, k] + T_curr[i-1, j, k]
                                            + T_curr[i, j+1, k]  + T_curr[i, j-1, k]
                                            + T_curr[i, j, k+1]  + T_curr[i, j, k-1] - 6.0 * T_curr[i, j, k]) + T_curr[i, j, k];
            return T_next           

    return T_curr

@njit(inline="always", fastmath=False)
def step_cache(T_in, T_out, r):
    nx, ny, nz = T_in.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            k_prev = T_in[i, j, 0]
            k_curr = T_in[i, j, 1]
            
            for k in range(1, nz - 1):
                k_next = T_in[i, j, k + 1]
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * k_curr + T_in[i-1, j, k]
                                     + T_in[i, j+1, k]  - 2.0 * k_curr + T_in[i, j-1, k]
                                     + k_next  - 2.0 * k_curr + k_prev) + k_curr;
                k_prev = k_curr
                k_curr = k_next

@njit
def solve_heat_numba_caching_1(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        # Explicit 3D Finite Difference via 3-level nested loops
        step_cache(T_curr, T_next, r)
        step_cache(T_next, T_curr, r)

    if (steps % 2 == 1):
        step_cache(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", fastmath=False)
def step_tiled_i(T_in, T_out, r, tile=64):
    nx, ny, nz = T_in.shape
    
    # Outer loops define the 3D tile coordinates (i0, j0, k0)
    for i0 in range(1, nx - 1, tile):
        i1 = min(i0 + tile, nx - 1)

        for i in range(i0, i1):
            for j in range(1, ny - 1):
                k_prev = T_in[i, j, 0]
                k_curr = T_in[i, j, 1]
            
                for k in range(1, nz -  1):
                    k_next = T_in[i, j, k + 1]
                    T_out[i, j, k] = r * (
                        T_in[i + 1, j, k] - 2.0 * k_curr + T_in[i - 1, j, k]
                        + T_in[i, j + 1, k] - 2.0 * k_curr + T_in[i, j - 1, k]
                        + k_next - 2.0 * k_curr + k_prev
                    ) + k_curr
                    
                    k_prev = k_curr
                    k_curr = k_next


@njit(fastmath=False)
def solve_heat_numba_tiling_i_16(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_i(T_curr, T_next, r, 16)
        step_tiled_i(T_next, T_curr, r, 16)

    if steps % 2 == 1:
        step_tiled_i(T_curr, T_next, r, 16)
        return T_next

    return T_curr

@njit(fastmath=False)
def solve_heat_numba_tiling_i_32(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_i(T_curr, T_next, r, 32)
        step_tiled_i(T_next, T_curr, r, 32)

    if steps % 2 == 1:
        step_tiled_i(T_curr, T_next, r, 32)
        return T_next

    return T_curr

@njit(fastmath=False)
def solve_heat_numba_tiling_i_64(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_i(T_curr, T_next, r)
        step_tiled_i(T_next, T_curr, r)

    if steps % 2 == 1:
        step_tiled_i(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", fastmath=False)
def step_tiled_j(T_in, T_out, r, tile_size=64):
    nx, ny, nz = T_in.shape
    
    for j0 in range(1, ny - 1, tile_size):
        j1 = min(j0 + tile_size, ny - 1)

        for i in range(1, nx - 1):
            for j in range(j0, j1):
                k_prev = T_in[i, j, 0]
                k_curr = T_in[i, j, 1]
            
                for k in range(1, nz -  1):
                    k_next = T_in[i, j, k + 1]
                    T_out[i, j, k] = r * (
                        T_in[i + 1, j, k] - 2.0 * k_curr + T_in[i - 1, j, k]
                        + T_in[i, j + 1, k] - 2.0 * k_curr + T_in[i, j - 1, k]
                        + k_next - 2.0 * k_curr + k_prev
                    ) + k_curr
                    
                    k_prev = k_curr
                    k_curr = k_next

@njit(fastmath=False)
def solve_heat_numba_tiling_j16(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_j(T_curr, T_next, r, 16)
        step_tiled_j(T_next, T_curr, r, 16)

    if steps % 2 == 1:
        step_tiled_j(T_curr, T_next, r, 16)
        return T_next

    return T_curr

@njit(fastmath=False)
def solve_heat_numba_tiling_j32(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_j(T_curr, T_next, r, 32)
        step_tiled_j(T_next, T_curr, r, 32)

    if steps % 2 == 1:
        step_tiled_j(T_curr, T_next, r, 32)
        return T_next

    return T_curr

@njit(fastmath=False)
def solve_heat_numba_tiling_j64(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_j(T_curr, T_next, r)
        step_tiled_j(T_next, T_curr, r)

    if steps % 2 == 1:
        step_tiled_j(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", fastmath=False)
def step_tiled_ij(T_in, T_out, r, tile=64):
    nx, ny, nz = T_in.shape
    
    # Outer loops define the 3D tile coordinates (i0, j0, k0)
    for i0 in range(1, nx - 1, tile):
        i1 = min(i0 + tile, nx - 1)
        for j0 in range(1, ny - 1, tile):
            j1 = min(j0 + tile, ny - 1)

            for i in range(i0, i1):
                for j in range(j0, j1):
                    k_prev = T_in[i, j, 0]
                    k_curr = T_in[i, j, 1]
                
                    for k in range(1, nz -  1):
                        k_next = T_in[i, j, k + 1]
                        T_out[i, j, k] = r * (
                            T_in[i + 1, j, k] - 2.0 * k_curr + T_in[i - 1, j, k]
                            + T_in[i, j + 1, k] - 2.0 * k_curr + T_in[i, j - 1, k]
                            + k_next - 2.0 * k_curr + k_prev
                        ) + k_curr
                        
                        k_prev = k_curr
                        k_curr = k_next


@njit(fastmath=False)
def solve_heat_numba_tiling_ij16(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_ij(T_curr, T_next, r, 16)
        step_tiled_ij(T_next, T_curr, r, 16)

    if steps % 2 == 1:
        step_tiled_ij(T_curr, T_next, r, 16)
        return T_next

    return T_curr

def solve_heat_numba_tiling_ij32(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_ij(T_curr, T_next, r, 32)
        step_tiled_ij(T_next, T_curr, r, 32)

    if steps % 2 == 1:
        step_tiled_ij(T_curr, T_next, r, 32)
        return T_next

    return T_curr

def solve_heat_numba_tiling_ij(T, steps, r):
    nx, ny, nz = T.shape
    
    T_curr = T.copy()
    T_next = T.copy()

    for _ in range(steps // 2):
        step_tiled_ij(T_curr, T_next, r)
        step_tiled_ij(T_next, T_curr, r)

    if steps % 2 == 1:
        step_tiled_ij(T_curr, T_next, r)
        return T_next

    return T_curr

# ---------------------------------------------------------
# Baseline 2: Copy Buffer
# ---------------------------------------------------------
@njit
def solve_heat_copy_buffer_2(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                         + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                         + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];


        # Swap buffers for next time step
        T_curr = T_next.copy()

    return T_curr

# ---------------------------------------------------------
# Baseline 3: Heat Equation in Numba with Fast Math
# ---------------------------------------------------------
@njit(fastmath=True)
def solve_heat_numba_fast_math_3(T, steps, r):
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

        if (steps % 2 == 1):
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        T_next[i, j, k] =  r * (T_curr[i+1, j, k] - 2.0 * T_curr[i, j, k] + T_curr[i-1, j, k]
                                            + T_curr[i, j+1, k]  - 2.0 * T_curr[i, j, k] + T_curr[i, j-1, k]
                                            + T_curr[i, j, k+1]  - 2.0 * T_curr[i, j, k] + T_curr[i, j, k-1]) + T_curr[i, j, k];
            return T_next           

    return T_curr

# ---------------------------------------------------------
# Baseline 4: Loop order change
# ---------------------------------------------------------
@njit(inline="always", fastmath=False)
def step_loop_jki(T_in, T_out, r):
    nx, ny, nz = T_in.shape
   
    for j in range(1, ny - 1):
        for k in range(1, nz - 1):
            for i in range(1, nx - 1):
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * T_in[i, j, k] + T_in[i-1, j, k]
                                    + T_in[i, j+1, k]  - 2.0 * T_in[i, j, k] + T_in[i, j-1, k]
                                    + T_in[i, j, k+1]  - 2.0 * T_in[i, j, k] + T_in[i, j, k-1]) + T_in[i, j, k];

@njit(fastmath=False)
def solve_heat_numba_loop_change_jki_4(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        step_loop_jki(T_curr, T_next, r)
        step_loop_jki(T_next, T_curr, r)

    if (steps % 2 == 1):
        step_loop_jki(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", fastmath=False)
def step_loop_kji(T_in, T_out, r):
    nx, ny, nz = T_in.shape
   
    for k in range(1, nz - 1):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * T_in[i, j, k] + T_in[i-1, j, k]
                                    + T_in[i, j+1, k]  - 2.0 * T_in[i, j, k] + T_in[i, j-1, k]
                                    + T_in[i, j, k+1]  - 2.0 * T_in[i, j, k] + T_in[i, j, k-1]) + T_in[i, j, k];

@njit(fastmath=False)
def solve_heat_numba_loop_change_kji(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        step_loop_kji(T_curr, T_next, r)
        step_loop_kji(T_next, T_curr, r)

    if (steps % 2 == 1):
        step_loop_kji(T_curr, T_next, r)
        return T_next

    return T_curr

# ---------------------------------------------------------
# Baseline 4: Heat Equation in Numba with Parallel
# ---------------------------------------------------------
@njit(inline="always", parallel=True)
def step_parallel_i(T_in, T_out, r):
    nx, ny, nz = T_in.shape
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            k_prev = T_in[i, j, 0]
            k_curr = T_in[i, j, 1]
            
            for k in range(1, nz - 1):
                k_next = T_in[i, j, k + 1]
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * k_curr + T_in[i-1, j, k]
                                     + T_in[i, j+1, k]  - 2.0 * k_curr + T_in[i, j-1, k]
                                     + k_next  - 2.0 * k_curr + k_prev) + k_curr;
                k_prev = k_curr
                k_curr = k_next

@njit(parallel=True, fastmath=False)
def solve_heat_numba_parallel_6(T, steps, r):
    nx, ny, nz = T.shape
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        step_parallel_i(T_curr, T_next, r)
        step_parallel_i(T_next, T_curr, r)

    if (steps % 2):
        step_parallel_i(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", parallel=True)
def step_parallel_ij(T_in, T_out, r):
    nx, ny, nz = T_in.shape
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            k_prev = T_in[i, j, 0]
            k_curr = T_in[i, j, 1]
            
            for k in range(1, nz - 1):
                k_next = T_in[i, j, k + 1]
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * k_curr + T_in[i-1, j, k]
                                     + T_in[i, j+1, k]  - 2.0 * k_curr + T_in[i, j-1, k]
                                     + k_next  - 2.0 * k_curr + k_prev) + k_curr;
                k_prev = k_curr
                k_curr = k_next

@njit(parallel=True, fastmath=False)
def solve_heat_numba_parallel_7(T, steps, r):
    nx, ny, nz = T.shape
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        step_parallel_ij(T_curr, T_next, r)
        step_parallel_ij(T_next, T_curr, r)

    if (steps % 2):
        step_parallel_ij(T_curr, T_next, r)
        return T_next

    return T_curr

@njit(inline="always", parallel=True)
def step_parallel_ijk(T_in, T_out, r):
    nx, ny, nz = T_in.shape
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            k_prev = T_in[i, j, 0]
            k_curr = T_in[i, j, 1]
            
            for k in prange(1, nz - 1):
                k_next = T_in[i, j, k + 1]
                T_out[i, j, k] =  r * (T_in[i+1, j, k] - 2.0 * k_curr + T_in[i-1, j, k]
                                     + T_in[i, j+1, k]  - 2.0 * k_curr + T_in[i, j-1, k]
                                     + k_next  - 2.0 * k_curr + k_prev) + k_curr;
                k_prev = k_curr
                k_curr = k_next

@njit(parallel=True, fastmath=True)
def solve_heat_numba_parallel_8(T, steps, r):
    nx, ny, nz = T.shape
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps//2):
        step_parallel_ijk(T_curr, T_next, r)
        step_parallel_ijk(T_next, T_curr, r)

    if (steps % 2):
        step_parallel_ijk(T_curr, T_next, r)
        return T_next

    return T_curr

@njit
def solve_heat_numba_vectorized_11(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps // 2):
        
        C = T_curr[1:-1, 1:-1, 1:-1]
        neighbors = (
            T_curr[2:,   1:-1, 1:-1] + T_curr[:-2,  1:-1, 1:-1] +  # X neighbors
            T_curr[1:-1, 2:,   1:-1] + T_curr[1:-1, :-2,  1:-1] +  # Y neighbors
            T_curr[1:-1, 1:-1, 2:  ] + T_curr[1:-1, 1:-1, :-2]     # Z neighbors
        )
        T_next[1:-1, 1:-1, 1:-1] = r * (neighbors - 6.0 * C) + C

        C_next = T_next[1:-1, 1:-1, 1:-1]
        neighbors_next = (
            T_next[2:,   1:-1, 1:-1] + T_next[:-2,  1:-1, 1:-1] +  # X neighbors
            T_next[1:-1, 2:,   1:-1] + T_next[1:-1, :-2,  1:-1] +  # Y neighbors
            T_next[1:-1, 1:-1, 2:  ] + T_next[1:-1, 1:-1, :-2]     # Z neighbors
        )
        T_curr[1:-1, 1:-1, 1:-1] = r * (neighbors_next - 6.0 * C_next) + C_next

    if (steps % 2):
        C = T_curr[1:-1, 1:-1, 1:-1]
        neighbors = (
            T_curr[2:,   1:-1, 1:-1] + T_curr[:-2,  1:-1, 1:-1] +  # X neighbors
            T_curr[1:-1, 2:,   1:-1] + T_curr[1:-1, :-2,  1:-1] +  # Y neighbors
            T_curr[1:-1, 1:-1, 2:  ] + T_curr[1:-1, 1:-1, :-2]     # Z neighbors
        )
        T_next[1:-1, 1:-1, 1:-1] = r * (neighbors - 6.0 * C) + C
        return T_next

    return T_curr


# Define the 3D 7-point stencil kernel
@stencil
def heat_kernel(arr, r):
    return r * (
        arr[1, 0, 0] - 2.0 * arr[0, 0, 0] + arr[-1, 0, 0] + 
        arr[0, 1, 0] - 2.0 * arr[0, 0, 0] + arr[0, -1, 0] +  
        arr[0, 0, 1] - 2.0 * arr[0, 0, 0] + arr[0, 0, -1]    
    ) + arr[0, 0, 0]

@njit(parallel=True, fastmath=False)
def solve_heat_numba_stencil(T, steps, r):
    nx, ny, nz = T.shape
    
    # Create working buffers
    T_curr = T.copy()
    T_next = np.zeros((nx, ny, nz), dtype=np.float64)

    for step in range(steps // 2):
        # Apply stencil with pre-allocated output buffers using 'out='
        heat_kernel(T_curr, r, out=T_next)
        heat_kernel(T_next, r, out=T_curr)

    return T_curr

# ---------------------------------------------------------
# Baseline Reference SciPy Laplace Solver
# ---------------------------------------------------------
def solve_heat_scipy(T, steps, r):
    T_curr = T.copy()
    d2T = np.empty_like(T_curr)

    for _ in range(steps):
        d2T = laplace(T_curr, output=d2T, mode='constant', cval=0.0)
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

    # Cannot run ground truth at reduced grid since memory is predominant 
    # bottleneck. Save defaults for faster testing
    # Compute ground truth reference solution for correctness verification
    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = N
    center = N//2
    half   = N_py // 2
    T_reduced =  T_init[
        center - half : center + half,
        center - half : center + half,
        center - half : center + half
    ].copy()

    T_expected = solve_heat_scipy(T_init, steps, r)
    total_flops_reduced = 1 * (N_py ** 3) * 13  #Do just one step for unoptimized program
    T_expected_reduced = solve_heat_scipy(T_reduced, 1, r)
    total_flops = steps * (N ** 3) * 13 

    def measure(fn, reps=3):
        # Correctness check against reference matrix T_expected
        T_result = fn(T_init, steps, r)
        is_correct = np.allclose(T_result, T_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            fn(T_init, steps, r)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9 

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python
    t0 = time.perf_counter()
    T_result = solve_heat_naive_python_0(T_reduced, 1, r)

    py_elapsed = time.perf_counter() - t0
    py_gflops  = (total_flops_reduced / py_elapsed) / 1e9 
    py_correct = np.allclose(T_result, T_expected_reduced, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_niave", solve_heat_numba_1),
        ("2_numba_reduced_op", solve_heat_numba_reduce_op_1),
        ("3_numba_fast_math", solve_heat_numba_fast_math_3),
        ("3_copy_buffer", solve_heat_copy_buffer_2),
        ("4_numba_cached", solve_heat_numba_caching_1),
        ("5_numba_tiling_i16", solve_heat_numba_tiling_i_16),
        ("5_numba_tiling_i32", solve_heat_numba_tiling_i_32),
        ("5_numba_tiling_i64", solve_heat_numba_tiling_i_64),
        ("6_numba_tiling_j16", solve_heat_numba_tiling_j16),
        ("6_numba_tiling_j32", solve_heat_numba_tiling_j32),
        ("6_numba_tiling_j64", solve_heat_numba_tiling_j64),
        ("6_numba_tiling_ij16", solve_heat_numba_tiling_ij16),
        ("6_numba_tiling_ij32", solve_heat_numba_tiling_ij32),
        ("6_numba_tiling_ij64", solve_heat_numba_tiling_ij),
        ("7_loop_change_jki", solve_heat_numba_loop_change_jki_4),
        ("8_loop_change_kji", solve_heat_numba_loop_change_kji),
        ("9_numba_parallel", solve_heat_numba_parallel_6),
        ("10_numba_parallel", solve_heat_numba_parallel_7),
        ("8_numba_parallel", solve_heat_numba_parallel_8),
        ("8_parallel_tiling", solve_heat_numba_parallel_8),
        ("11_vectorized", solve_heat_numba_vectorized_11),
        ("13_stencil_kernel", solve_heat_numba_stencil),
        ("13_sci_py_laplacian", solve_heat_scipy),
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
    N_input = 512
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
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Time (s)':<12} |{'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
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

        print(f"| {name:<33} | {gflops:10.3f} | {elapsed:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
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
    plt.title(f"Numba Heat 3D Benchmark Performance on Grid {N_input}x{N_input}x{N_input}")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    # save the plot as a PNG file
    plt.savefig(img_path, dpi=300)