# EVOLVE-BLOCK-START

import numpy as np
import time
import sys

N = 26

def _compute_score(pos, radii):
    """Compute the sum of radii, the objective to maximize."""
    return np.sum(radii)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _fix_validity(pos, radii):
    """
    Minimal repair to ensure strict validity.
    Shrinks radii slightly if constraints are violated.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Boundary check
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r * 0.999999
    
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                # Scale down both radii to fit
                if dist > 1e-10:
                    scale = dist / sum_r * 0.999999
                    radii[i] *= scale
                    radii[j] *= scale
                else:
                    radii[i] *= 0.5
                    radii[j] *= 0.5
                    
    return pos, radii

def _init_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.09
            k += 1
    pos, radii = _fix_validity(pos, radii)
    return pos, radii

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    dx = 0.2
    dy = dx * np.sqrt(3) / 2
    
    k = 0
    for row in range(5):
        for col in range(5):
            if k >= N:
                break
            x = 0.1 + col * dx + (row % 2) * dx / 2
            y = 0.1 + row * dy
            if x < 0.9 and y < 0.9:
                pos[k] = [x, y]
                radii[k] = 0.09
                k += 1
    
    rng = np.random.RandomState(123)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        pos[k] = [x, y]
        radii[k] = 0.08
        k += 1
        
    pos, radii = _fix_validity(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.11, N)
    pos, radii = _fix_validity(pos, radii)
    return pos, radii

def _solve_cvxpy(pos_init, radii_init, deadline):
    """Solve using CVXPY SOCP."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
        
    x = cp.Variable((N, 2))
    r = cp.Variable(N)
    
    objective = cp.Maximize(cp.sum(r))
    
    constraints = [
        r >= 0,
        x[:, 0] - r >= 0,
        1 - x[:, 0] - r >= 0,
        x[:, 1] - r >= 0,
        1 - x[:, 1] - r >= 0,
    ]
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
            
    prob = cp.Problem(objective, constraints)
    
    # Warm start
    try:
        x.value = pos_init
        r.value = radii_init
    except:
        pass
        
    solvers = []
    try:
        solvers.append(cp.CLARABEL)
    except:
        pass
    try:
        solvers.append(cp.SCS)
    except:
        pass
    try:
        solvers.append(cp.ECOS)
    except:
        pass
        
    if not solvers:
        return None

    for solver in solvers:
        try:
            prob.solve(solver=solver, warm_start=True)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                pos = np.array(x.value)
                radii = np.array(r.value)
                radii = np.maximum(radii, 1e-6)
                pos, radii = _fix_validity(pos, radii)
                if _is_valid(pos, radii):
                    return pos, radii
        except Exception:
            continue
            
    return None

def _solve_slsqp_convex(pos_init, radii_init, deadline):
    """
    Solve using SLSQP. 
    Note: For a fixed topology, the feasible region is convex. 
    SLSQP can navigate this if initialized well.
    """
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Boundary constraints
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        # Overlap constraints
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 300, 'ftol': 1e-10})
        p = res.x[:2*N].reshape(N, 2)
        r = res.x[2*N:]
        r = np.maximum(r, 1e-6)
        p, r = _fix_validity(p, r)
        if _is_valid(p, r):
            return p, r
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # Generate initializations
    inits = []
    pos, radii = _init_grid()
    inits.append((pos, radii))
    
    pos, radii = _init_hexagonal()
    inits.append((pos, radii))
    
    for seed in range(10):
        pos, radii = _init_random(seed)
        inits.append((pos, radii))

    # Strategy 1: Try CVXPY (True Convex Solver)
    use_cvxpy = True
    try:
        import cvxpy
        use_cvxpy = True
    except ImportError:
        use_cvxpy = False

    if use_cvxpy:
        for pos, radii in inits:
            if time.time() > deadline - 10:
                break
            res = _solve_cvxpy(pos, radii, deadline)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p
                    best_radii = r

    # Strategy 2: Fallback or Refinement with SLSQP (Convex per topology)
    # If CVXPY worked, we might still try SLSQP on the best solution to polish?
    # Or if CVXPY failed, use SLSQP on all inits.
    # To save time, if CVXPY gave a good result, we skip SLSQP on others unless time permits.
    
    if best_score < 0 or not use_cvxpy:
        # CVXPY not used or failed, run SLSQP on all
        for pos, radii in inits:
            if time.time() > deadline - 5:
                break
            res = _solve_slsqp_convex(pos, radii, deadline)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p
                    best_radii = r
    else:
        # CVXPY succeeded, try to polish the best solution with SLSQP if time permits
        if time.time() < deadline - 5:
             res = _solve_slsqp_convex(best_pos, best_radii, deadline)
             if res:
                 p, r = res
                 s = np.sum(r)
                 if s > best_score:
                     best_score = s
                     best_pos = p
                     best_radii = r

    if best_pos is None:
        # Final fallback
        pos, radii = _init_grid()
        pos, radii = _fix_validity(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
