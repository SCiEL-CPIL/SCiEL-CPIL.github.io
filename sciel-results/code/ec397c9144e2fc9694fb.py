"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-6):
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii, tol=1e-7):
    """
    Simple relaxation to fix small overlaps and boundary violations.
    """
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        # Check boundaries
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    if dist < 1e-9:
                        # Move them apart randomly
                        pos[i, 0] += 0.01
                        pos[j, 0] -= 0.01
                    else:
                        # Push apart
                        push = (sum_r - dist + tol) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                    valid = False
        
        # Clip to unit square
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, 1)
        
        if valid:
            break
    return pos, radii

def _energy(pos, radii):
    """
    Energy function for Simulated Annealing.
    Lower is better.
    """
    base = -np.sum(radii)
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        r = radii[i]
        x, y = pos[i]
        penalty += max(0.0, r - x)**2
        penalty += max(0.0, r - (1.0 - x))**2
        penalty += max(0.0, r - y)**2
        penalty += max(0.0, r - (1.0 - y))**2
    
    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = (radii[i] + radii[j]) - dist
            if overlap > 0:
                penalty += overlap * overlap
                
    return base + 100.0 * penalty

def _simulated_annealing(pos_init, radii_init, time_budget=10.0, seed=42):
    """
    Runs Simulated Annealing to improve the packing.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    pos = pos_init.copy()
    radii = radii_init.copy()
    
    current_energy = _energy(pos, radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_energy = current_energy
    best_valid_score = -np.inf
    
    # Check if initial is valid
    if _is_valid(pos, radii):
        best_valid_score = np.sum(radii)
    
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.9995
    
    steps = 0
    while time.time() - start_time < time_budget:
        # Propose a move
        i = rng.randint(N)
        
        # Perturbation sizes scale with T
        delta_pos = rng.normal(0, T, 2)
        delta_r = rng.normal(0, T * 0.5)
        
        new_pos = pos.copy()
        new_radii = radii.copy()
        
        new_pos[i, 0] += delta_pos[0]
        new_pos[i, 1] += delta_pos[1]
        new_radii[i] = max(1e-5, radii[i] + delta_r)
        
        # Clip to unit square
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0, 1)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0, 1)
        
        new_energy = _energy(new_pos, new_radii)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or rng.rand() < np.exp(-delta_e / T):
            pos = new_pos
            radii = new_radii
            current_energy = new_energy
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
                best_radii = radii.copy()
                
            # Track best valid score specifically
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_valid_score:
                    best_valid_score = score
                    # We keep best_pos/best_radii as the energy-minimizer, 
                    # but if we find a valid one with higher sum of radii, 
                    # it's likely better. 
                    # Actually, if it's valid, penalty is 0, so energy = -sum(r).
                    # So minimizing energy is equivalent to maximizing sum(r) for valid configs.
                    # So best_energy logic is sufficient.
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        steps += 1
        
        # Periodic restart if stuck? No, just keep going.
        
    return best_pos, best_radii

def _initial_greedy_best_fit():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(8000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.95
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _objective(x):
    radii = x[2*N:]
    return -np.sum(radii)

def _all_constraints(x):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Boundary constraints
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    # Overlap constraints
    o_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            o_cons.append(dist_sq - sum_r * sum_r)
    o_cons = np.array(o_cons)
    
    return np.concatenate([b_cons, o_cons])

def _final_polish(pos, radii, time_budget=10.0):
    """
    High-precision SLSQP polish.
    """
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, pos[i,0] - 0.05), min(1.0, pos[i,0] + 0.05)))
        bounds.append((max(0.0, pos[i,1] - 0.05), min(1.0, pos[i,1] + 0.05)))
    for i in range(N):
        bounds.append((max(1e-5, radii[i] * 0.9), min(0.5, radii[i] * 1.1)))
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': _all_constraints},
                          options={'maxiter': 200, 'ftol': 1e-12})
        if result.success or result.fun < _objective(x0):
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-9)
            if _is_valid(pos_pp, radii_pp, tol=1e-8):
                pos, radii = pos_pp, radii_pp
    except Exception:
        pass
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Initializers
    inits = [
        _initial_greedy_best_fit,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
    ]
    
    for idx, init_func in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
            
        # Run Simulated Annealing
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        sa_budget = min(20.0, remaining * 0.5)
        
        if sa_budget > 5.0:
            pos_sa, radii_sa = _simulated_annealing(pos_init, rad_init, time_budget=sa_budget, seed=idx+1)
            
            # Post-process SA result
            pos_pp, radii_pp = _post_process(pos_sa, radii_sa)
            
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            
            # Also keep the raw SA result if valid (post-process might shrink radii too much?)
            if _is_valid(pos_sa, radii_sa):
                score_raw = np.sum(radii_sa)
                if score_raw > best_score:
                    best_score = score_raw
                    best_pos = pos_sa.copy()
                    best_radii = radii_sa.copy()
                    
        # Check initial if SA didn't improve or time ran out
        if _is_valid(pos_init, rad_init):
            score_init = np.sum(rad_init)
            if score_init > best_score:
                best_score = score_init
                best_pos = pos_init.copy()
                best_radii = rad_init.copy()
                
    # Final Polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_polished, radii_polished = _final_polish(best_pos, best_radii, time_budget=10.0)
        if _is_valid(pos_polished, radii_polished, tol=1e-8):
            score_polished = np.sum(radii_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pos = pos_polished.copy()
                best_radii = radii_polished.copy()

    if best_pos is None or not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
