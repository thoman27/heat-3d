import taichi as ti
import numpy as np
import time

# Initialize Taichi (using GPU backend like Vulkan, CUDA, or ROCm)
ti.init(arch=ti.gpu)

N = 256
reps = 3
steps = 100
dx = dy = dz = 0.01      # in meters
dt = 0.001               # in seconds
alpha = 1.17e-4          # Thermal diffusivity (m^2/s) - Copper
r = alpha * dt / (dx**2) # CFL Stability Criterion for 3D: r <= 1/6 (~0.1667)

# Allocate Taichi fields
field_a = ti.field(dtype=ti.f64, shape=(N, N, N))
field_b = ti.field(dtype=ti.f64, shape=(N, N, N))

@ti.kernel
def heat_step(curr: ti.template(), next_gen: ti.template(), r: ti.f64):
    # Using ti.template() allows passing fields dynamically into the kernel
    for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (1, N - 1)):
        next_gen[i, j, k] = r * (
            curr[i+1, j, k] - 2.0 * curr[i, j, k] + curr[i-1, j, k] +
            curr[i, j+1, k] - 2.0 * curr[i, j, k] + curr[i, j-1, k] +
            curr[i, j, k+1] - 2.0 * curr[i, j, k] + curr[i, j, k-1]
        ) + curr[i, j, k]

def solve_heat_taichi(T_host, steps, r):
    # Load initial host data into field_a
    field_a.from_numpy(np.ascontiguousarray(T_host, dtype=np.float64))
    field_b.from_numpy(np.ascontiguousarray(T_host, dtype=np.float64))
    
    curr = field_a
    next_gen = field_b
    
    for step in range(steps):
        heat_step(curr, next_gen, r)
        curr, next_gen = next_gen, curr
        
    return curr.to_numpy()

T_init = np.zeros((N, N, N), dtype=np.float64)

# Place a hot central source
cx, cy, cz = N // 2, N // 2, N // 2
for i in range(N):
    for j in range(N):
        for k in range(N):
            if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 <= 16:
                T_init[i, j, k] = 100.0

total_flops = steps * (N ** 3) * 13 
start = time.perf_counter()
for _ in range(reps):
    solve_heat_taichi(T_init, steps, r)
elapsed = (time.perf_counter() - start) / reps

gflops = (total_flops / elapsed) / 1e9 
print(f"|         | {'GFLOP/s':<10}  | {'Time (s)':<12}")
print(f"| Results | {gflops:10.3f}  | {elapsed:10.3f} ")