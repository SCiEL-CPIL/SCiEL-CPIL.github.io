# EVOLVE-BLOCK-START
import time
import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    t0 = time.time()
    DEADLINE = 600.0

    def elapsed():
        return time.time() - t0

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # Use a fixed, high-resolution grid for volume estimation
    # 40x40x40 = 64,000 points is fast enough for many iterations
    # and accurate enough for optimization
    n_grid = 40
    a_grid, b_grid, c_grid = np.meshgrid(
        np.linspace(0, 1, n_grid), 
        np.linspace(0, 1, n_grid), 
        np.linspace(0, 1, n_grid), 
        indexing='ij'
    )
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()
    c_flat = c_grid.flatten()
    
    def estimate_volume(positions):
        """
        positions: (64, 2) array of base (x, y)
        Returns: estimated volume of union in [0,1]^3
        """
        x_base = positions[:, 0]
        y_base = positions[:, 1]
        
        # Tube k occupies [x_k + c*i_k/N, x_k + w + c*i_k/N] x [y_k + c*j_k/N, y_k + w + c*j_k/N]
        # We check if point (a, b, c) is inside tube k:
        # x_base[k] + c_flat * i_flat[k] / N <= a <= x_base[k] + w + c_flat * i_flat[k] / N
        # y_base[k] + c_flat * j_flat[k] / N <= b <= y_base[k] + w + c_flat * j_flat[k] / N
        
        # Vectorized check:
        # (64, 64000) arrays might be large, let's do it in chunks or rely on BLAS
        # 64 * 64000 * 8 bytes * 4 arrays = ~130 MB, acceptable
        
        x_shift = np.outer(x_base, np.ones_like(c_flat)) + np.outer(i_flat / N, c_flat)
        y_shift = np.outer(y_base, np.ones_like(c_flat)) + np.outer(j_flat / N, c_flat)
        
        in_x = (a_flat[None, :] >= x_shift) & (a_flat[None, :] <= x_shift + w)
        in_y = (b_flat[None, :] >= y_shift) & (b_flat[None, :] <= y_shift + w)
        
        in_tube = in_x & in_y  # (64, 64000)
        
        # Union: any tube contains the point
        covered = np.any(in_tube, axis=0)
        
        # Volume in [0,1]^3
        return np.sum(covered) / len(covered)

    # Initial positions: simple grid
    # x in [0, 7/8], y in [0, 7/8]
    x0 = np.arange(N) * w
    y0 = np.arange(N) * w
    x_init, y_init = np.meshgrid(x0, y0, indexing='ij')
    init_pos = np.stack([x_init.flatten(), y_init.flatten()], axis=1)
    
    best_pos = init_pos.copy()
    best_vol = estimate_volume(best_pos)
    
    # Simulated Annealing
    # We want to keep the union within a small box to minimize volume.
    # The evaluator scales by box area. So we should minimize (union_volume_in_box / box_area).
    # But the MC estimate in the code returns fraction * (X1-X0)*(Y1-Y0).
    # If we constrain positions to [0, 1], the box is [0, 1]x[0,1] (roughly), area ~1.
    # Actually, the box is determined by min/max of positions.
    # To be safe and simple, let's assume we optimize for the raw volume in [0,1]^3,
    # but we must account for the box scaling in the final score.
    # Score = 1 / V. V = fraction * Area.
    # If we keep points in [0,1], Area is roughly 1.
    # Let's try to minimize the volume directly, keeping positions in [0, 1-w].
    
    rng = np.random.default_rng(42)
    T = 0.05
    T_min = 0.001
    alpha = 0.995
    
    n_iter = 0
    while elapsed() < DEADLINE - 30:
        # Perturb a few random positions
        n_perturb = rng.integers(1, 5)
        idx = rng.choice(64, size=n_perturb, replace=False)
        
        new_pos = best_pos.copy()
        # Gaussian step
        step = rng.normal(0, T, size=(n_perturb, 2))
        new_pos[idx] += step
        
        # Clamp to [0, 1-w] to keep box small
        new_pos = np.clip(new_pos, 0.0, 1.0 - w)
        
        new_vol = estimate_volume(new_pos)
        
        # Delta E
        delta = new_vol - best_vol
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_pos = new_pos
            best_vol = new_vol
            
        T *= alpha
        if T < T_min:
            T = T_min
            
        n_iter += 1
        
        # Periodic restart with smaller T? Or just keep going.
        # If we get stuck, we can restart from a slightly different seed.
        # But for now, let's just run.
        
        if n_iter % 1000 == 0:
            # print(f"Iter {n_iter}, Vol {best_vol:.6f}, T {T:.6f}")
            pass

    # Final refinement: try to shrink the box if possible?
    # The volume formula is V = fraction * Area.
    # If we shift all points, the box shifts but area stays same.
    # If we scale points, area changes.
    # But the shears are fixed in absolute units (i/8).
    # So scaling positions scales the box but not the shears relative to box?
    # No, shears are added to positions.
    # If we scale positions by S, box scales by S. Shears stay same.
    # This might increase or decrease overlap.
    # Given the complexity, we stick to the optimized positions.
    
    return best_pos

# EVOLVE-BLOCK-END
