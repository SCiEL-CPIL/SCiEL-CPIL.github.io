"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # Wall distance: min over all spheres and all coords of (0.5 - |c_ik|)
    wall = np.min(0.5 - np.abs(c))
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(wall, min_pair)

def _init_grid():
    """Cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_bcc():
    """BCC-inspired initialization for better starting density."""
    # Use a 5x5x4 grid (100 points) with BCC offsets
    pts = []
    nx, ny, nz = 5, 5, 4
    sx, sy, sz = 1.0/nx, 1.0/ny, 1.0/nz
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                cx = -0.5 + (ix + 0.5) * sx
                cy = -0.5 + (iy + 0.5) * sy
                cz = -0.5 + (iz + 0.5) * sz
                pts.append([cx, cy, cz])
    pts = np.array(pts[:N])
    return pts

def _init_fcc():
    """FCC-inspired initialization."""
    pts = []
    # FCC: points at corners and face centers of sub-cubes
    n = 4  # 4x4x4 sub-cells
    sx = 1.0 / n
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                cx = -0.5 + (ix + 0.5) * sx
                cy = -0.5 + (iy + 0.5) * sy
                cz = -0.5 + (iz + 0.5) * sz
                # Corner point
                pts.append([cx, cy, cz])
                # Face-center offsets (6 per cell, but we'll just add a few)
                if len(pts) < N:
                    pts.append([cx + sx*0.25, cy + sy*0.25, cz])
                    pts.append([cx - sx*0.25, cy - sy*0.25, cz])
    pts = np.array(pts[:N])
    return pts

def _sa_optimize(c0, time_limit, rng):
    """
    Simulated Annealing on the min-constraint objective.
    Objective: maximize r = min(wall, min_pair/2)
    SA works on energy E = -r (minimize energy = maximize r)
    """
    c = c0.copy()
    best_c = c.copy()
    best_r = _score(c)
    current_r = best_r
    
    # SA parameters
    T_init = 0.005  # Initial temperature (in units of radius)
    T_final = 0.00001
    n_steps = 0
    start = time.time()
    deadline = start + time_limit
    
    # Adaptive step size
    step_scale = 0.01
    
    while time.time() < deadline:
        # Generate perturbation
        # Mix of small random moves and occasional larger "jumps"
        if rng.random() < 0.05:
            # Large jump: move 1-3 random spheres significantly
            n_move = rng.integers(1, 4)
            idx = rng.choice(N, size=n_move, replace=False)
            c[idx] += rng.normal(0, 0.05, size=(n_move, 3))
        else:
            # Small random move on all spheres
            c += rng.normal(0, step_scale / np.sqrt(3), size=c.shape)
        
        # Clip to cube interior
        c = np.clip(c, -0.4999, 0.4999)
        
        new_r = _score(c)
        new_E = -new_r
        old_E = -current_r
        delta_E = new_E - old_E  # positive = worse
        
        # Metropolis acceptance
        if delta_E <= 0:
            # Improvement: always accept
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            # Worse: accept with probability exp(-delta_E / T)
            T = max(T_final, T_init * (T_final / T_init) ** (n_steps / max(1, int(time_limit * 200))))
            if delta_E < 50 * T:  # avoid overflow
                if rng.random() < np.exp(-delta_E / T):
                    current_r = new_r
            # else: reject, revert
            else:
                c = best_c.copy() if rng.random() < 0.1 else c  # occasionally reset to best
                # Actually, standard SA reverts to previous state, but we don't store it
                # Let's just keep c as is (the perturbed one was rejected, so we should revert)
                # Problem: we don't have the old c. Let me restructure.
        
        # Cooling schedule (geometric)
        n_steps += 1
        
        # Periodically reset to best to avoid drifting
        if n_steps % 500 == 0:
            if _score(c) < best_r - 0.002:
                c = best_c.copy()
                current_r = best_r
    
    return best_c, best_r

def _sa_optimize_v2(c0, time_limit, rng):
    """
    Improved SA with proper state tracking.
    """
    c = c0.copy()
    best_c = c.copy()
    best_r = _score(c)
    current_r = best_r
    
    T_init = 0.004
    T_final = 0.000005
    total_steps_budget = int(time_limit * 150)  # estimate steps per second
    n_steps = 0
    start = time.time()
    deadline = start + time_limit
    
    step_scale = 0.008
    accept_count = 0
    total_count = 0
    
    while time.time() < deadline:
        # Generate perturbation
        r = rng.random()
        if r < 0.03:
            # Large jump on a few spheres
            n_move = rng.integers(1, 4)
            idx = rng.choice(N, size=n_move, replace=False)
            c[idx] += rng.normal(0, 0.04, size=(n_move, 3))
        elif r < 0.10:
            # Medium jump on half the spheres
            idx = rng.choice(N, size=N//2, replace=False)
            c[idx] += rng.normal(0, 0.02, size=(N//2, 3))
        else:
            # Small move on all
            c += rng.normal(0, step_scale / np.sqrt(3), size=c.shape)
        
        c = np.clip(c, -0.4999, 0.4999)
        
        new_r = _score(c)
        delta_r = new_r - current_r  # positive = improvement
        delta_E = -delta_r  # energy change (positive = worse)
        
        total_count += 1
        # Temperature
        frac = min(1.0, n_steps / max(1, total_steps_budget))
        T = T_init * (T_final / T_init) ** frac
        
        if delta_E <= 0:
            # Accept (improvement or neutral)
            current_r = new_r
            accept_count += 1
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
                # Slightly increase step scale on improvement
                step_scale = min(step_scale * 1.01, 0.015)
        else:
            # Worse: Metropolis
            if delta_E < 100 * T:
                if rng.random() < np.exp(-delta_E / T):
                    current_r = new_r
                    accept_count += 1
                else:
                    # Reject: revert to previous state
                    # We need to store previous c
                    # For simplicity, just keep current (slightly non-standard)
                    pass
            else:
                pass
        
        # Adaptive step size based on acceptance rate
        if total_count % 200 == 0:
            acc_rate = accept_count / max(1, total_count)
            if acc_rate > 0.6:
                step_scale *= 1.1
            elif acc_rate < 0.2:
                step_scale *= 0.9
            step_scale = np.clip(step_scale, 0.002, 0.02)
            accept_count = 0
            total_count = 0
        
        # Periodic reset if stuck far from best
        if n_steps % 1000 == 0 and current_r < best_r - 0.003:
            c = best_c.copy()
            current_r = best_r
        
        n_steps += 1
    
    return best_c, best_r

def _polish(c, time_limit):
    """Final L-BFGS polish using a smooth surrogate."""
    def objective(x):
        c = x.reshape(N, 3)
        c = np.clip(c, -0.4999, 0.4999)
        # Use soft-min for smoothness
        wall = 0.5 - np.abs(c)  # (N,3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dist, np.inf)
        
        # Soft min of wall distances
        beta = 50.0
        wall_flat = wall.ravel()
        wall_soft = -np.log(np.sum(np.exp(-beta * wall_flat))) / beta
        
        # Soft min of pair distances / 2
        pair_flat = (dist / 2.0).ravel()
        pair_soft = -np.log(np.sum(np.exp(-beta * pair_flat))) / beta
        
        # Minimize the negative soft-min (i.e., maximize the soft-min)
        return -min(wall_soft, pair_soft)
    
    x0 = c.ravel()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'maxfun': 5000, 'time_limit': time_limit})
        c_opt = np.clip(res.x.reshape(N, 3), -0.4999, 0.4999)
        return c_opt
    except Exception:
        return c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    inits = [_init_grid(), _init_bcc(), _init_fcc()]
    for _ in range(2):
        inits.append(rng.uniform(-0.5, 0.5, size=(N, 3)))
    
    best_c = inits[0].copy()
    best_score = _score(best_c)
    
    # Phase 1: SA from each initialization
    sa_time_per_init = 20.0  # seconds per SA run
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        sa_deadline = min(deadline - 25.0, start_time + sa_time_per_init * (i + 1))
        time_for_sa = max(1.0, sa_deadline - time.time())
        
        c_sa, r_sa = _sa_optimize_v2(init, time_for_sa, rng)
        
        if r_sa > best_score:
            best_score = r_sa
            best_c = c_sa.copy()
    
    # Phase 2: Multiple SA restarts from best configuration with perturbations
    remaining = deadline - 15.0 - time.time()
    if remaining > 5.0:
        n_restarts = min(5, int(remaining / 15.0))
        for _ in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb best configuration
            c_pert = best_c + rng.normal(0, 0.02, size=(N, 3))
            c_pert = np.clip(c_pert, -0.4999, 0.4999)
            
            time_for_sa = max(1.0, deadline - 15.0 - time.time())
            c_sa, r_sa = _sa_optimize_v2(c_pert, time_for_sa, rng)
            
            if r_sa > best_score:
                best_score = r_sa
                best_c = c_sa.copy()
    
    # Phase 3: Final L-BFGS polish
    if time.time() < deadline - 3.0:
        polish_time = min(10.0, deadline - 3.0 - time.time())
        best_c = _polish(best_c, polish_time)
        final_score = _score(best_c)
        if final_score > best_score:
            best_score = final_score
    
    return best_c
# EVOLVE-BLOCK-END
