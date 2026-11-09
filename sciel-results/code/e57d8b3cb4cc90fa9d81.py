# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=500000):
    """Compute the union volume using the exact Monte Carlo method from the task."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Get i and j indices for each tube
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Compute box bounds
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if each point is inside at least one tube
    # Tube k at height c: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(64):
        i_k = i_idx[k]
        j_k = j_idx[k]
        x_k = x[k]
        y_k = y[k]
        
        x_left = x_k + c * i_k / N
        x_right = x_left + 1.0 / N
        y_left = y_k + c * j_k / N
        y_right = y_left + 1.0 / N
        
        in_x = (a >= x_left) & (a <= x_right)
        in_y = (b >= y_left) & (b <= y_right)
        inside |= (in_x & in_y)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _compute_volume_fast(positions, seed=12345, n_samples=50000):
    """Faster volume computation with fewer samples for optimization."""
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(64):
        i_k = i_idx[k]
        j_k = j_idx[k]
        x_k = x[k]
        y_k = y[k]
        
        x_left = x_k + c * i_k / N
        x_right = x_left + 1.0 / N
        y_left = y_k + c * j_k / N
        y_right = y_left + 1.0 / N
        
        in_x = (a >= x_left) & (a <= x_right)
        in_y = (b >= y_left) & (b <= y_right)
        inside |= (in_x & in_y)
    
    fraction = np.mean(inside)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    return V


def _pairwise_overlap_volume(positions):
    """
    Compute sum of pairwise intersection volumes between tubes.
    This is a deterministic measure that correlates with union volume.
    """
    N = 8
    x = positions[:, 0]
    y = positions[:, 1]
    
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    total_overlap = 0.0
    
    for k1 in range(64):
        for k2 in range(k1 + 1, 64):
            i1, j1 = i_idx[k1], j_idx[k1]
            i2, j2 = i_idx[k2], j_idx[k2]
            
            # Tube k1 at height c: x in [x1 + c*i1/N, x1 + c*i1/N + 1/N], y in [y1 + c*j1/N, y1 + c*j1/N + 1/N]
            # Tube k2 at height c: x in [x2 + c*i2/N, x2 + c*i2/N + 1/N], y in [y2 + c*j2/N, y2 + c*j2/N + 1/N]
            
            # For a given c, the x-intervals overlap if:
            # max(x1 + c*i1/N, x2 + c*i2/N) < min(x1 + c*i1/N + 1/N, x2 + c*i2/N + 1/N)
            # This is equivalent to |x1 - x2 + c*(i1-i2)/N| < 1/N
            
            # Similarly for y: |y1 - y2 + c*(j1-j2)/N| < 1/N
            
            dx = x[k1] - x[k2] + (i1 - i2) / N * np.linspace(0, 1, 1000)
            dy = y[k1] - y[k2] + (j1 - j2) / N * np.linspace(0, 1, 1000)
            
            # Count where both x and y intervals overlap
            x_overlap = np.abs(dx) < 1.0 / N
            y_overlap = np.abs(dy) < 1.0 / N
            
            overlap_fraction = np.mean(x_overlap & y_overlap)
            total_overlap += overlap_fraction
    
    return total_overlap


def _admm_optimize(positions, max_time=550, n_samples=30000):
    """
    ADMM-based optimization using pairwise overlap as a proxy for union volume.
    We minimize sum of pairwise overlaps, which is a convex-like surrogate.
    """
    start_time = time.time()
    best = positions.copy()
    best_V = _compute_volume_fast(best, n_samples=n_samples)
    
    N = 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # ADMM parameters
    rho = 1.0  # penalty parameter
    max_iter = 2000
    
    current = positions.copy()
    
    # Precompute for speed
    i_diff = np.zeros((64, 64))
    j_diff = np.zeros((64, 64))
    for k1 in range(64):
        for k2 in range(64):
            i_diff[k1, k2] = (i_idx[k1] - i_idx[k2]) / N
            j_diff[k1, k2] = (j_idx[k1] - j_idx[k2]) / N
    
    # ADMM loop
    for it in range(max_iter):
        if time.time() - start_time > max_time - 30:
            break
        
        # Update step size based on iteration
        step = 0.01 / (1.0 + it * 0.01)
        
        # For each tube, compute gradient of pairwise overlap
        grad_x = np.zeros(64)
        grad_y = np.zeros(64)
        
        for k1 in range(64):
            for k2 in range(k1 + 1, 64):
                i1, j1 = i_idx[k1], j_idx[k1]
                i2, j2 = i_idx[k2], j_idx[k2]
                
                # Approximate gradient using finite differences
                # Overlap volume between tube k1 and k2 as function of positions
                # dx = x1 - x2 + c*(i1-i2)/N, dy = y1 - y2 + c*(j1-j2)/N
                # Overlap exists when |dx| < 1/N and |dy| < 1/N
                # The overlap volume is integral over c of indicator(|dx(c)|<1/N)*indicator(|dy(c)|<1/N)
                
                # For small changes, we can approximate the gradient
                dx0 = current[k1, 0] - current[k2, 0] + (i1 - i2) / N * 0.5  # average c
                dy0 = current[k1, 1] - current[k2, 1] + (j1 - j2) / N * 0.5
                
                # Gradient contribution (approximate)
                if abs(dx0) < 1.0/N:
                    grad_x[k1] += 1.0 / (N * (1.0 + 10*abs(dx0)))
                    grad_x[k2] -= 1.0 / (N * (1.0 + 10*abs(dx0)))
                
                if abs(dy0) < 1.0/N:
                    grad_y[k1] += 1.0 / (N * (1.0 + 10*abs(dy0)))
                    grad_y[k2] -= 1.0 / (N * (1.0 + 10*abs(dy0)))
        
        # ADMM update
        current[:, 0] -= step * grad_x
        current[:, 1] -= step * grad_y
        
        # Keep positions in reasonable bounds
        current = np.clip(current, -2, 3)
        
        # Evaluate
        if it % 10 == 0:
            current_V = _compute_volume_fast(current, n_samples=n_samples)
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
    
    return best, best_V


def _simulated_annealing(positions, max_time=550, n_samples=20000):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    best = positions.copy()
    best_V = _compute_volume_fast(best, n_samples=n_samples)
    
    current = positions.copy()
    current_V = best_V
    
    T = 0.1  # initial temperature
    cooling = 0.999
    min_T = 0.001
    
    rng = np.random.default_rng(99)
    
    while T > min_T and time.time() - start_time < max_time - 30:
        # Generate random perturbation
        trial = current + rng.normal(0, 0.05, size=current.shape)
        trial = np.clip(trial, -2, 3)
        
        trial_V = _compute_volume_fast(trial, n_samples=n_samples)
        
        delta = trial_V - current_V
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = trial
            current_V = trial_V
            
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
        
        T *= cooling
    
    return best, best_V


def _hybrid_optimize(positions, max_time=550):
    """
    Combine ADMM, simulated annealing, and coordinate descent.
    """
    start_time = time.time()
    best = positions.copy()
    best_V = _compute_volume_fast(best, n_samples=30000)
    
    # Phase 1: ADMM optimization
    if time.time() - start_time < max_time - 100:
        try:
            admm_result, admm_V = _admm_optimize(positions, max_time=max_time - 100)
            if admm_V < best_V:
                best = admm_result.copy()
                best_V = admm_V
        except Exception:
            pass
    
    # Phase 2: Simulated annealing from best
    if time.time() - start_time < max_time - 50:
        try:
            sa_result, sa_V = _simulated_annealing(best, max_time=max_time - 50)
            if sa_V < best_V:
                best = sa_result.copy()
                best_V = sa_V
        except Exception:
            pass
    
    # Phase 3: Final coordinate descent polish
    if time.time() - start_time < max_time - 20:
        try:
            current = best.copy()
            current_V = best_V
            step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
            rng = np.random.default_rng(77)
            
            for step in step_sizes:
                if time.time() - start_time > max_time - 20:
                    break
                for _ in range(50):
                    if time.time() - start_time > max_time - 20:
                        break
                    improved = False
                    for k in range(64):
                        if time.time() - start_time > max_time - 20:
                            break
                        for coord in range(2):
                            for direction in [-1, 1]:
                                trial = current.copy()
                                trial[k, coord] += direction * step * rng.uniform(0.5, 1.5)
                                if trial[k, coord] < -2 or trial[k, coord] > 3:
                                    continue
                                trial_V = _compute_volume_fast(trial, n_samples=20000)
                                if trial_V < current_V - 1e-10:
                                    current = trial
                                    current_V = trial_V
                                    improved = True
                                    if current_V < best_V:
                                        best = current.copy()
                                        best_V = current_V
        except Exception:
            pass
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses hybrid optimization: ADMM + simulated annealing + coordinate descent.
    """
    N = 8
    
    # Keich binary construction for n = 2^k base positions
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Deadline guard
    start_time = time.time()
    max_total_time = 620  # Leave 10s margin
    
    # Try hybrid optimization if we have time
    if time.time() - start_time < max_total_time - 100:
        try:
            refined, _ = _hybrid_optimize(positions, max_time=max_total_time - 20)
            positions = refined
        except Exception:
            pass  # Keep original if optimization fails
    
    return positions


# EVOLVE-BLOCK-END
