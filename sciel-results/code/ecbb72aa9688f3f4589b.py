# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_volume(positions, seed=12345, n_samples=500000):
    """Compute the union volume using the exact Monte Carlo method from the task."""
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


def _compute_volume_very_fast(positions, seed=42, n_samples=5000):
    """Very fast volume estimate for inner loop of Nelder-Mead."""
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


def _nelder_mead_2d(points, max_time=30.0, n_samples=8000, seed=42):
    """
    Nelder-Mead optimization over a subset of points (2D coordinates).
    points: list of (row, col) tuples to optimize.
    """
    start_time = time.time()
    n = len(points)
    
    if n == 0:
        return None, float('inf')
    
    # Build initial simplex
    positions = np.array(points, dtype=np.float64).copy()
    # positions is (n, 2)
    
    # Create simplex: n+1 vertices, each is a full 128-dim array
    # We'll represent the simplex as a list of 128-dim arrays
    # Base position
    base = np.zeros(128)
    for idx, (row, col) in enumerate(points):
        base[2*row] = positions[idx, 0]
        base[2*row+1] = positions[idx, 1]
    
    # Simplex vertices
    simplex = [base.copy()]
    step_size = 0.1
    for idx in range(n):
        for col in [0, 1]:
            vert = base.copy()
            vert[2*points[idx][0] + col] += step_size
            simplex.append(vert)
    
    # Evaluate all vertices
    def eval_vertex(vertex):
        pos = np.zeros((64, 2))
        for idx, (row, col) in enumerate(points):
            pos[row, 0] = vertex[2*row]
            pos[row, 1] = vertex[2*row+1]
        # Fill in other positions from base
        for r in range(64):
            if r not in [p[0] for p in points]:
                pos[r, 0] = base[2*r]
                pos[r, 1] = base[2*r+1]
        return _compute_volume_very_fast(pos, n_samples=n_samples, seed=seed)
    
    # Actually, we need to keep the full positions array and only modify the subset
    full_pos = np.zeros((64, 2))
    for r in range(64):
        full_pos[r, 0] = base[2*r]
        full_pos[r, 1] = base[2*r+1]
    
    def eval_full(pos):
        return _compute_volume_very_fast(pos, n_samples=n_samples, seed=seed)
    
    # Simplex as list of full (64,2) position arrays
    simplex = [full_pos.copy()]
    for idx in range(n):
        for col in [0, 1]:
            vert = full_pos.copy()
            vert[points[idx][0], col] += step_size
            simplex.append(vert)
    
    # Evaluate
    values = [eval_full(v) for v in simplex]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    max_iter = 200
    
    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < 1e-10:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_worst = simplex[-1]
        x_reflect = centroid + alpha * (centroid - x_worst)
        f_reflect = eval_full(x_reflect)
        
        if f_reflect < values[0]:
            # Expansion
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = eval_full(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                values[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                values[-1] = f_reflect
        elif f_reflect < values[-2]:
            simplex[-1] = x_reflect
            values[-1] = f_reflect
        else:
            # Contraction
            if f_reflect < values[-1]:
                x_contract = centroid + rho * (x_reflect - centroid)
                f_contract = eval_full(x_contract)
            else:
                x_contract = centroid - rho * (x_worst - centroid)
                f_contract = eval_full(x_contract)
            
            if f_contract < min(f_reflect, values[-1]):
                simplex[-1] = x_contract
                values[-1] = f_contract
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = eval_full(simplex[i])
    
    # Return best
    best_idx = np.argmin(values)
    best_pos = simplex[best_idx].copy()
    best_V = values[best_idx]
    
    return best_pos, best_V


def _block_optimization(positions, max_time=120.0, n_samples=8000, seed=42, block_size=4):
    """
    Optimize blocks of 2x2 tubes (4 positions) using Nelder-Mead.
    This is a structural change: instead of optimizing individual coordinates,
    we optimize 2x2 blocks which have correlated movement patterns.
    """
    start_time = time.time()
    N = 8
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    # Define 2x2 blocks in the 8x8 grid
    blocks = []
    for bi in range(0, N, block_size):
        for bj in range(0, N, block_size):
            block_points = []
            for di in range(block_size):
                for dj in range(block_size):
                    i = bi + di
                    j = bj + dj
                    if i < N and j < N:
                        k = i * N + j
                        block_points.append(k)
            blocks.append(block_points)
    
    rng = np.random.default_rng(seed)
    
    # Round-robin over blocks
    block_order = list(range(len(blocks)))
    
    while time.time() - start_time < max_time:
        improved = False
        rng.shuffle(block_order)
        
        for b_idx in block_order:
            if time.time() - start_time > max_time:
                break
            
            block = blocks[b_idx]
            if len(block) < 2:
                continue
            
            # Extract current positions for this block
            block_positions = current[block, :].copy()
            
            # Run Nelder-Mead on this block
            time_remaining = max_time - (time.time() - start_time)
            time_per_block = min(15.0, time_remaining / max(1, len(block_order)))
            
            refined, V_ref = _nelder_mead_2d(block, max_time=time_per_block, n_samples=n_samples, seed=seed)
            
            if refined is not None:
                trial = current.copy()
                trial[block, :] = refined[block, :]
                
                # Validate
                if np.all(np.isfinite(trial)):
                    trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
        
        if not improved:
            # Perturb to escape local minimum
            perturbation = rng.normal(0, 0.03, size=current.shape)
            current += perturbation
            current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
    
    return best, best_V


def _row_column_optimization(positions, max_time=90.0, n_samples=8000, seed=42):
    """
    Optimize row-by-row and column-by-column.
    Each row has 8 tubes with the same i but different j.
    Each column has 8 tubes with the same j but different i.
    """
    start_time = time.time()
    N = 8
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    
    # Row indices: for each i, the 8 tubes (i, j) for j=0..7
    # Column indices: for each j, the 8 tubes (i, j) for i=0..7
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Optimize each row
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            row_tubes = [i * N + j for j in range(N)]
            row_positions = current[row_tubes, :].copy()
            
            time_remaining = max_time - (time.time() - start_time)
            time_per = min(5.0, time_remaining / max(1, N))
            
            refined, V_ref = _nelder_mead_2d(row_tubes, max_time=time_per, n_samples=n_samples, seed=seed)
            
            if refined is not None:
                trial = current.copy()
                trial[row_tubes, :] = refined[row_tubes, :]
                
                if np.all(np.isfinite(trial)):
                    trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
        
        # Optimize each column
        for j in range(N):
            if time.time() - start_time > max_time:
                break
            
            col_tubes = [i * N + j for i in range(N)]
            col_positions = current[col_tubes, :].copy()
            
            time_remaining = max_time - (time.time() - start_time)
            time_per = min(5.0, time_remaining / max(1, N))
            
            refined, V_ref = _nelder_mead_2d(col_tubes, max_time=time_per, n_samples=n_samples, seed=seed)
            
            if refined is not None:
                trial = current.copy()
                trial[col_tubes, :] = refined[col_tubes, :]
                
                if np.all(np.isfinite(trial)):
                    trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
        
        if not improved:
            perturbation = rng.normal(0, 0.02, size=current.shape)
            current += perturbation
            current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
    
    return best, best_V


def _diagonal_optimization(positions, max_time=60.0, n_samples=8000, seed=42):
    """
    Optimize along diagonals of the 8x8 grid.
    Diagonal d: tubes (i, j) where j - i = d mod 8.
    """
    start_time = time.time()
    N = 8
    current = positions.copy()
    current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
    best = current.copy()
    best_V = current_V
    
    rng = np.random.default_rng(seed)
    
    # Define diagonals
    diagonals = []
    for d in range(N):
        diag = []
        for i in range(N):
            j = (i + d) % N
            k = i * N + j
            diag.append(k)
        diagonals.append(diag)
    
    while time.time() - start_time < max_time:
        improved = False
        
        for d_idx in range(N):
            if time.time() - start_time > max_time:
                break
            
            diag = diagonals[d_idx]
            time_remaining = max_time - (time.time() - start_time)
            time_per = min(5.0, time_remaining / max(1, N))
            
            refined, V_ref = _nelder_mead_2d(diag, max_time=time_per, n_samples=n_samples, seed=seed)
            
            if refined is not None:
                trial = current.copy()
                trial[diag, :] = refined[diag, :]
                
                if np.all(np.isfinite(trial)):
                    trial_V = _compute_volume_fast(trial, n_samples=n_samples, seed=seed)
                    
                    if trial_V < current_V - 1e-12:
                        current = trial
                        current_V = trial_V
                        improved = True
                        
                        if current_V < best_V:
                            best = current.copy()
                            best_V = current_V
        
        if not improved:
            perturbation = rng.normal(0, 0.02, size=current.shape)
            current += perturbation
            current_V = _compute_volume_fast(current, n_samples=n_samples, seed=seed)
            if current_V < best_V:
                best = current.copy()
                best_V = current_V
    
    return best, best_V


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


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initialization, then refines with
    block-based Nelder-Mead optimization, row/column/diagonal optimization,
    and final local search.
    """
    N = 8
    
    initial_positions = _keich_positions(N)
    
    start_time = time.time()
    max_total_time = 580  # Safety margin before 630s limit
    
    best_positions = initial_positions.copy()
    best_V = _compute_volume_fast(initial_positions, n_samples=50000, seed=12345)
    
    # Phase 1: Block optimization (2x2 blocks with Nelder-Mead)
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 120:
        refined, V_ref = _block_optimization(best_positions, max_time=100.0, n_samples=8000, seed=42, block_size=4)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 2: Row/column optimization
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 90:
        refined, V_ref = _row_column_optimization(best_positions, max_time=80.0, n_samples=8000, seed=55)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 3: Diagonal optimization
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 60:
        refined, V_ref = _diagonal_optimization(best_positions, max_time=50.0, n_samples=8000, seed=66)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 4: Local search refinement
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 60:
        refined, V_ref = _local_search(best_positions, max_time=45.0, n_samples=12000, seed=1, start_step=0.05)
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 5: Multi-start with block optimization
    rng = np.random.default_rng(123)
    time_remaining = max_total_time - (time.time() - start_time)
    num_starts = min(3, max(1, int(time_remaining / 40)))
    
    for i in range(num_starts):
        if time.time() - start_time > max_total_time - 50:
            break
        
        noise = rng.normal(0, 0.04, size=best_positions.shape)
        start_pos = best_positions + noise
        start_pos[:, 0] = np.clip(start_pos[:, 0], -1.0, 2.5)
        start_pos[:, 1] = np.clip(start_pos[:, 1], -1.0, 2.5)
        
        time_per_start = min(30.0, (max_total_time - 50 - (time.time() - start_time)) / max(1, num_starts - i))
        refined, V_ref = _block_optimization(start_pos, max_time=time_per_start, n_samples=8000, seed=200+i, block_size=4)
        
        if V_ref < best_V:
            best_positions = refined
            best_V = V_ref
    
    # Phase 6: Final refinement
    time_remaining = max_total_time - (time.time() - start_time)
    if time_remaining > 20:
        final_time = min(30.0, time_remaining - 5)
        if final_time > 10:
            refined, V_ref = _local_search(best_positions, max_time=final_time, n_samples=20000, seed=999, start_step=0.02)
            if V_ref < best_V:
                best_positions = refined
                best_V = V_ref
    
    return best_positions


# EVOLVE-BLOCK-END
