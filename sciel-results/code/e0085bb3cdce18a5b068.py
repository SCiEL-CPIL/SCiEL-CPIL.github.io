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
    """Keich binary construction for base positions."""
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
    return np.stack([base_x, base_y], axis=1)


def _parametric_positions(params, N=8):
    """
    Generate positions from a low-dimensional parametric model.
    params: array of shape (n_params,)
    
    Model: x_k = base_x[i] + sum_m a_m * cos(2*pi*m*i/N) * exp(-d_m)
           y_k = base_y[j] + sum_m b_m * cos(2*pi*m*j/N) * exp(-d_m)
    
    where d_m encodes decay with mode number.
    """
    base = _keich_positions(N)
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    n_modes = 3
    # params layout: [a_0, a_1, a_2, b_0, b_1, b_2, shift_x, shift_y, scale_x, scale_y]
    a_coefs = params[:n_modes]
    b_coefs = params[n_modes:2*n_modes]
    shift_x = params[2*n_modes]
    shift_y = params[2*n_modes + 1]
    scale_x = params[2*n_modes + 2]
    scale_y = params[2*n_modes + 3]
    
    x_pos = base[:, 0].copy()
    y_pos = base[:, 1].copy()
    
    for m in range(n_modes):
        freq = (m + 1)
        x_mod = a_coefs[m] * np.cos(2 * np.pi * freq * i_idx / N) * np.exp(-0.3 * m)
        y_mod = b_coefs[m] * np.cos(2 * np.pi * freq * j_idx / N) * np.exp(-0.3 * m)
        x_pos += x_mod
        y_pos += y_mod
    
    x_pos = x_pos * scale_x + shift_x
    y_pos = y_pos * scale_y + shift_y
    
    return np.stack([x_pos, y_pos], axis=1)


def _local_search(positions, max_time=60.0, n_samples=15000, seed=42, start_step=0.08):
    """
    Perform local search using coordinate descent with adaptive step sizes.
    """
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    N = 8
    rng = np.random.default_rng(seed)
    
    step = start_step
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(64)
        
        for k in indices:
            if time.time() - start_time > max_time:
                break
                
            for coord in [0, 1]:
                for direction in [1, -1]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1]:
                        actual_step = step * step_mult
                        trial = current.copy()
                        trial[k, coord] += direction * actual_step
                        
                        if trial[k, coord] < -1.5 or trial[k, coord] > 3.0:
                            continue
                        
                        trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                        
                        if trial_V < current_V - 1e-12:
                            current = trial
                            current_V = trial_V
                            improved = True
                            
                            if current_V < best_V:
                                best = current.copy()
                                best_V = current_V
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            
                            step *= 0.9
                            break
                    else:
                        continue
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                step = max(0.04, step * 1.5)
                no_improve_count = 0
                perturbation = rng.normal(0, 0.04, size=current.shape)
                current += perturbation
                current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
                if current_V < best_V:
                    best = current.copy()
                    best_V = current_V
            else:
                step *= 0.8
                if step < 0.003:
                    step = 0.015
    
    return best, best_V


def _simulated_annealing(positions, max_time=120.0, n_samples=15000, seed=77):
    """Simulated annealing for escaping local minima."""
    start_time = time.time()
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    
    T = 0.08
    T_min = 0.002
    cooling = 0.994
    
    while time.time() - start_time < max_time and T > T_min:
        trial = current.copy()
        n_perturb = rng.integers(2, 6)
        indices = rng.choice(64, size=n_perturb, replace=False)
        for k in indices:
            trial[k, 0] += rng.normal(0, 0.06)
            trial[k, 1] += rng.normal(0, 0.06)
        
        trial[:, 0] = np.clip(trial[:, 0], -1.5, 3.0)
        trial[:, 1] = np.clip(trial[:, 1], -1.5, 3.0)
        
        trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
        
        if trial_V < current_V:
            current = trial
            current_V = trial_V
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
        else:
            delta = trial_V - current_V
            if delta < 0.02 and np.log(rng.uniform()) > -delta / T:
                current = trial
                current_V = trial_V
        
        T *= cooling
    
    return best, best_V


def _parametric_optimize(max_time=60.0, seed=100):
    """
    Optimize the parametric model parameters, then map to full positions.
    """
    start_time = time.time()
    n_modes = 3
    n_params = 2 * n_modes + 4  # a_coefs, b_coefs, shift_x, shift_y, scale_x, scale_y
    
    rng = np.random.default_rng(seed)
    
    # Initial params: small deformations
    params = np.zeros(n_params)
    params[2*n_modes] = 0.0   # shift_x
    params[2*n_modes + 1] = 0.0  # shift_y
    params[2*n_modes + 2] = 1.0  # scale_x
    params[2*n_modes + 3] = 1.0  # scale_y
    
    positions = _parametric_positions(params)
    current_V = _compute_volume_fast(positions, n_samples=10000, seed=seed)
    best_params = params.copy()
    best_V = current_V
    best_positions = positions.copy()
    
    step = 0.05
    no_improve = 0
    
    while time.time() - start_time < max_time:
        improved = False
        param_order = rng.permutation(n_params)
        
        for p in param_order:
            if time.time() - start_time > max_time:
                break
            
            for direction in [1, -1]:
                for mult in [1.0, 0.5, 0.25]:
                    trial_params = params.copy()
                    trial_params[p] += direction * step * mult
                    
                    # Keep scale positive
                    if p >= 2*n_modes + 2 and trial_params[p] < 0.5:
                        continue
                    
                    trial_pos = _parametric_positions(trial_params)
                    trial_V = _compute_volume_fast(trial_pos, n_samples=10000, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        params = trial_params
                        current_V = trial_V
                        positions = trial_pos
                        improved = True
                        
                        if current_V < best_V:
                            best_params = params.copy()
                            best_V = current_V
                            best_positions = positions.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        step *= 0.9
                        break
                else:
                    continue
                break
        
        if not improved:
            no_improve += 1
            if no_improve >= 8:
                step = max(0.02, step * 1.5)
                no_improve = 0
                params += rng.normal(0, 0.02, size=n_params)
                if params[2*n_modes+2] < 0.5: params[2*n_modes+2] = 0.5
                if params[2*n_modes+3] < 0.5: params[2*n_modes+3] = 0.5
                positions = _parametric_positions(params)
                current_V = _compute_volume_fast(positions, n_samples=10000, seed=seed)
                if current_V < best_V:
                    best_params = params.copy()
                    best_V = current_V
                    best_positions = positions.copy()
            else:
                step *= 0.85
                if step < 0.005:
                    step = 0.01
    
    return best_positions, best_V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with
    parametric optimization, multi-phase local search and simulated annealing.
    """
    N = 8
    
    initial_positions = _keich_positions(N)
    
    start_time = time.time()
    max_total_time = 580  # Safety margin before 630s limit
    
    best_positions = initial_positions.copy()
    best_V = _compute_volume_fast(initial_positions, n_samples=50000, seed=12345)
    
    # Phase 1: Parametric optimization (low-dimensional search)
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 80:
        refined, V_ref = _parametric_optimize(max_time=50.0, seed=100)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 2: Local search from the best position
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 60:
        refined, V_ref = _local_search(best_positions, max_time=45.0, n_samples=12000, seed=1, start_step=0.08)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 3: Simulated annealing
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 80:
        refined, V_ref = _simulated_annealing(best_positions, max_time=60.0, n_samples=12000, seed=55)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 4: Multi-start local search
    rng = np.random.default_rng(123)
    time_remaining = max_total_time - (time.time() - start_time)
    num_starts = min(5, max(1, int(time_remaining / 30)))
    
    for i in range(num_starts):
        if time.time() - start_time > max_total_time - 50:
            break
        
        noise = rng.normal(0, 0.05, size=best_positions.shape)
        start_pos = best_positions + noise
        start_pos[:, 0] = np.clip(start_pos[:, 0], -1.0, 2.5)
        start_pos[:, 1] = np.clip(start_pos[:, 1], -1.0, 2.5)
        
        time_per_start = min(25.0, (max_total_time - 50 - (time.time() - start_time)) / max(1, num_starts - i))
        refined, V_ref = _local_search(start_pos, max_time=time_per_start, n_samples=15000, seed=200+i, start_step=0.05)
        
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 5: Final refinement
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 20:
        final_time = min(35.0, time_remaining - 5)
        if final_time > 10:
            refined, V_ref = _local_search(best_positions, max_time=final_time, n_samples=20000, seed=999, start_step=0.02)
            if V_ref < best_V:
                best_positions = refined
                best_V = V_ref
    
    return best_positions


# EVOLVE-BLOCK-END
