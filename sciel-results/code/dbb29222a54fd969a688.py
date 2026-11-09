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


def _compute_volume_fast(positions, seed=42, n_samples=20000):
    """Faster volume estimate for local search iterations."""
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


def _keich_positions(N=8):
    """Generate Keich binary construction positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _parametric_to_positions(params, N=8):
    """
    Convert low-dimensional parametric representation to 64x2 positions.
    
    params structure:
    - params[0:4]: global offsets (x0, y0, sx, sy)
    - params[4:10]: x Fourier modes (cos/sin for 3 frequencies)
    - params[10:16]: y Fourier modes (cos/sin for 3 frequencies)
    - params[16:22]: cross-coupling x modes
    - params[22:28]: cross-coupling y modes
    - params[28:34]: radial RBF coefficients (6 centers)
    
    Total: 34 parameters vs 128 independent coordinates.
    """
    N = 8
    x0, y0, sx, sy = params[0:4]
    
    # Grid indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # Normalize to [0, 1]
    inorm = i_flat / (N - 1)
    jnorm = j_flat / (N - 1)
    
    # Base position from Keich construction
    keich_x = _keich_positions(N)[i_flat.astype(int)]
    keich_y = _keich_positions(N)[j_flat.astype(int)]
    
    # Fourier deformation for x
    fx = np.zeros(64)
    for m in range(3):
        freq = (m + 1) * 2 * np.pi
        fx += params[4 + 2*m] * np.cos(freq * inorm)
        fx += params[5 + 2*m] * np.sin(freq * inorm)
    
    # Fourier deformation for y
    fy = np.zeros(64)
    for m in range(3):
        freq = (m + 1) * 2 * np.pi
        fy += params[10 + 2*m] * np.cos(freq * jnorm)
        fy += params[11 + 2*m] * np.sin(freq * jnorm)
    
    # Cross-coupling: x depends on j, y depends on i
    cx = np.zeros(64)
    for m in range(3):
        freq = (m + 1) * 2 * np.pi
        cx += params[16 + 2*m] * np.cos(freq * jnorm)
        cx += params[17 + 2*m] * np.sin(freq * jnorm)
    
    cy = np.zeros(64)
    for m in range(3):
        freq = (m + 1) * 2 * np.pi
        cy += params[22 + 2*m] * np.cos(freq * inorm)
        cy += params[23 + 2*m] * np.sin(freq * inorm)
    
    # Radial basis functions for fine-tuning
    rbf_centers_x = np.linspace(0, 1, 6)
    rbf_centers_y = np.linspace(0, 1, 6)
    rbf = np.zeros(64)
    for ci in range(6):
        for cj in range(6):
            dist_sq = (inorm - rbf_centers_x[ci])**2 + (jnorm - rbf_centers_y[cj])**2
            rbf += params[28 + ci*6 + cj] * np.exp(-dist_sq / 0.05)
    
    # Combine
    x_pos = keich_x * sx + x0 + fx + cx + 0.5 * rbf
    y_pos = keich_y * sy + y0 + fy + cy + 0.5 * rbf
    
    return np.stack([x_pos, y_pos], axis=1)


def _parametric_local_search(params, max_time=30.0, n_samples=15000, seed=42):
    """
    Local search in the low-dimensional parametric space.
    """
    start_time = time.time()
    current = params.copy()
    current_V = _compute_volume_fast(_parametric_to_positions(current), n_samples=n_samples)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    step = 0.05
    dim = len(params)
    
    while time.time() - start_time < max_time:
        improved = False
        
        for _ in range(50):  # Try 50 random coordinate moves per iteration
            if time.time() - start_time > max_time:
                break
            
            coord = rng.integers(0, dim)
            direction = rng.choice([-1, 1])
            actual_step = step * rng.uniform(0.5, 1.5)
            
            trial = current.copy()
            trial[coord] += direction * actual_step
            
            # Keep parameters in reasonable range
            if abs(trial[coord]) > 2.0:
                continue
            
            trial_pos = _parametric_to_positions(trial)
            trial_V = _compute_volume_fast(trial_pos, n_samples=n_samples)
            
            if trial_V < current_V - 1e-10:
                current = trial
                current_V = trial_V
                improved = True
                
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V
                
                step *= 0.95
                break
        
        if not improved:
            step *= 0.7
            if step < 0.001:
                step = 0.03
                perturbation = rng.normal(0, 0.01, size=dim)
                current = current + perturbation
                current_V = _compute_volume_fast(_parametric_to_positions(current), n_samples=n_samples)
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V
    
    return best, best_V


def _simulated_annealing(params, max_time=30.0, n_samples=15000, seed=42):
    """
    Simulated annealing in the parametric space.
    """
    start_time = time.time()
    current = params.copy()
    current_pos = _parametric_to_positions(current)
    current_V = _compute_volume_fast(current_pos, n_samples=n_samples)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    T = 0.1
    cooling = 0.95
    dim = len(params)
    
    while time.time() - start_time < max_time and T > 0.001:
        for _ in range(20):
            if time.time() - start_time > max_time:
                break
            
            # Random perturbation
            delta = rng.normal(0, 0.05, size=dim)
            trial = current + delta
            
            # Clip to reasonable range
            trial = np.clip(trial, -2.0, 2.0)
            
            trial_pos = _parametric_to_positions(trial)
            trial_V = _compute_volume_fast(trial_pos, n_samples=n_samples)
            
            delta_V = trial_V - current_V
            
            if delta_V < 0 or rng.random() < np.exp(-delta_V / T):
                current = trial
                current_V = trial_V
                
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V
        
        T *= cooling
    
    return best, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses a parametric deformation field approach: represent the 64 base positions
    via ~34 smooth parameters (Fourier modes + RBF) instead of 128 independent
    coordinates. This enforces global coherence and reduces the search space.
    """
    N = 8
    start_time = time.time()
    max_total_time = 620  # Safety margin before 630s limit
    
    # Initial parametric representation: Keich construction with zero deformations
    # params: [x0, y0, sx, sy, fx(6), fy(6), cx(6), cy(6), rbf(36)]
    initial_params = np.zeros(34)
    initial_params[0] = 0.0  # x offset
    initial_params[1] = 0.0  # y offset
    initial_params[2] = 1.0  # x scale
    initial_params[3] = 1.0  # y scale
    
    initial_positions = _parametric_to_positions(initial_params)
    
    # Verify initial volume
    best_V = _compute_volume_fast(initial_positions, n_samples=50000)
    best_params = initial_params.copy()
    
    # Phase 1: Local search in parametric space
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 40:
        phase1_time = min(40.0, time_remaining - 10)
        refined_params, V_ref = _parametric_local_search(
            initial_params, max_time=phase1_time, n_samples=15000, seed=1
        )
        if V_ref < best_V:
            best_params = refined_params
            best_V = V_ref
    
    # Phase 2: Simulated annealing from multiple starts
    rng = np.random.default_rng(123)
    time_remaining = max_total_time - (time.time() - start_time)
    num_sa_starts = 4
    time_per_sa = (time_remaining - 30) / num_sa if time_remaining > 30 else 0
    
    for i in range(num_sa_starts):
        if time.time() - start_time > max_total_time - 30:
            break
        
        # Perturb the best params
        noise = rng.normal(0, 0.03, size=34)
        start_params = best_params + noise
        start_params = np.clip(start_params, -2.0, 2.0)
        
        sa_time = min(time_per_sa, 30.0)
        if sa_time < 5:
            continue
        
        sa_params, V_sa = _simulated_annealing(
            start_params, max_time=sa_time, n_samples=15000, seed=100 + i
        )
        
        if V_sa < best_V:
            best_params = sa_params
            best_V = V_sa
    
    # Phase 3: Final local search refinement
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 20:
        final_time = min(25.0, time_remaining - 5)
        if final_time > 5:
            refined_params, V_ref = _parametric_local_search(
                best_params, max_time=final_time, n_samples=20000, seed=999
            )
            if V_ref < best_V:
                best_params = refined_params
                best_V = V_ref
    
    # Convert best params to positions
    best_positions = _parametric_to_positions(best_params)
    
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        # Fallback to initial
        best_positions = _parametric_to_positions(initial_params)
    
    return best_positions


# EVOLVE-BLOCK-END
