# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=50.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    # = m - (1/beta) * log(sum(exp(-beta * shifted)))
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _epigraph_objective(x, beta, penalty_lambda=50.0):
    """
    Epigraph formulation: minimize -t subject to t <= areas[i] for all i.
    Uses soft constraints via log-sum-exp.
    x: (2N+1,) array: first 2N are u,v coords, last is t.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty + 100.0 * t
        
    # Soft constraint: t <= min(areas) approximately
    # Use: -t + (1/beta)*log(sum(exp(-beta*(areas - t))))
    shifted = areas - t
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    return -t + (1.0 / beta) * log_sum_exp + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _local_optimize_nm(x0, bounds, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, bounds, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _simulated_annealing(x0, deadline, max_steps=5000, t0=0.1, t_end=1e-6):
    """
    Simulated annealing with topological swaps and local jitter.
    """
    rng = np.random.RandomState(789)
    current = x0.copy()
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    
    # Cooling schedule
    alpha = (t_end / t0) ** (1.0 / max_steps)
    temperature = t0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        neighbor = current.copy()
        move_type = rng.random()
        
        if move_type < 0.3:
            # Topological swap: swap barycentric coords of two random points
            i, j = rng.choice(N, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j].copy(), neighbor[i].copy()
        elif move_type < 0.7:
            # Local jitter: Gaussian perturbation of one point
            i = rng.randint(N)
            noise = rng.normal(0, 0.05, size=2)
            neighbor[i] += noise
            neighbor[i] = np.clip(neighbor[i], 0, None)
            s = neighbor[i, 0] + neighbor[i, 1]
            if s > 1.0:
                neighbor[i] /= s
        else:
            # Global jitter: perturb multiple points slightly
            n_pts = rng.randint(2, 5)
            idxs = rng.choice(N, size=n_pts, replace=False)
            for i in idxs:
                noise = rng.normal(0, 0.03, size=2)
                neighbor[i] += noise
                neighbor[i] = np.clip(neighbor[i], 0, None)
                s = neighbor[i, 0] + neighbor[i, 1]
                if s > 1.0:
                    neighbor[i] /= s
        
        neighbor = _project_to_bary_simplex(neighbor)
        neighbor_score = _min_area_ratio_from_bary(neighbor)
        
        if neighbor_score < 0:
            continue
            
        # Acceptance criterion
        if neighbor_score > current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            # Accept worse solution with probability exp((new - old) / T)
            delta = neighbor_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-10)):
                current = neighbor
                current_score = neighbor_score
        
        temperature *= alpha
        
    return best, best_score

def _final_polish(x0, bounds, time_budget=30.0):
    """
    Final high-precision polish: SLSQP with very tight tolerances,
    then a second SLSQP pass, then Nelder-Mead with adaptive.
    """
    start = time.time()
    current = x0.copy()
    best_score = _min_area_ratio_from_bary(current.reshape(N, 2))
    
    # Pass 1: SLSQP with tight tolerance
    if time.time() - start < time_budget * 0.4:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 2: SLSQP again from improved point
    if time.time() - start < time_budget * 0.7:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 3: Nelder-Mead with very tight tolerance
    if time.time() - start < time_budget * 0.9:
        try:
            res = minimize(
                _objective_slsqp, current, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    # Pass 4: Final SLSQP
    if time.time() - start < time_budget:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            cand = _project_to_bary_simplex(res.x.reshape(N, 2))
            sc = _min_area_ratio_from_bary(cand)
            if sc > best_score:
                best_score = sc
                current = cand.flatten()
        except Exception:
            pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts from Inspiration
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    bary5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.0, 0.3],
        [0.0, 0.7],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary5).flatten())
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 1: Local optimization from structured inits
    for init in inits:
        if time_left() < 5:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            best_local_x = init_bary
            best_local_score = _min_area_ratio_from_bary(init_bary)
            
            # Strategy: Multi-method refinement
            # 1. SLSQP
            if time_left() > 10:
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=200)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                if score_slsqp > best_local_score:
                    best_local_score = score_slsqp
                    best_local_x = bary_slsqp
                    init_bary = bary_slsqp

            # 2. Nelder-Mead
            if time_left() > 10:
                x_nm = _local_optimize_nm(init_bary.flatten(), bounds, maxiter=300)
                bary_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score_nm = _min_area_ratio_from_bary(bary_nm)
                if score_nm > best_local_score:
                    best_local_score = score_nm
                    best_local_x = bary_nm
                    init_bary = bary_nm

            # 3. L-BFGS-B with Smooth Objective (annealing)
            if time_left() > 10:
                current_x = init_bary.flatten()
                betas = [50.0, 100.0, 300.0, 800.0, 2000.0]
                for beta in betas:
                    if time_left() < 5:
                        break
                    res = minimize(
                        _objective_smooth,
                        current_x,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    x_opt = res.x.reshape(N, 2)
                    x_opt = _project_to_bary_simplex(x_opt)
                    score = _min_area_ratio_from_bary(x_opt)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_x = x_opt
                        current_x = x_opt.flatten()

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    # Phase 1.5: Simulated Annealing from best solution
    if time_left() > 30 and best_x is not None:
        try:
            sa_deadline = time.time() + min(25.0, time_left() * 0.5)
            sa_best, sa_score = _simulated_annealing(
                best_x.copy(), sa_deadline, max_steps=3000, t0=0.05, t_end=1e-6
            )
            if sa_score > best_score:
                best_score = sa_score
                best_x = sa_best
        except Exception:
            pass

    # Phase 2: Differential Evolution for global search
    if time_left() > 25:
        try:
            de_bounds = [(0, 1)] * (2 * N)
            
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = _project_to_bary_simplex(bary)
                score = _min_area_ratio_from_bary(bary)
                if score < 0:
                    return 1.0  # Penalty for invalid
                return -score
            
            # Run differential evolution with limited population and iterations
            res_de = differential_evolution(
                de_objective,
                de_bounds,
                seed=42,
                maxiter=25,
                popsize=12,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            
            if res_de.success or res_de.fun < -best_score:
                de_bary = _project_to_bary_simplex(res_de.x.reshape(N, 2))
                de_score = _min_area_ratio_from_bary(de_bary)
                if de_score > best_score:
                    best_score = de_score
                    best_x = de_bary
                    
        except Exception:
            pass

    # Phase 2.5: Second SA pass from DE result if improved
    if time_left() > 20 and best_x is not None:
        try:
            sa_deadline2 = time.time() + min(20.0, time_left() * 0.4)
            sa_best2, sa_score2 = _simulated_annealing(
                best_x.copy(), sa_deadline2, max_steps=2000, t0=0.03, t_end=1e-7
            )
            if sa_score2 > best_score:
                best_score = sa_score2
                best_x = sa_best2
        except Exception:
            pass

    # Phase 3: Random restarts with SLSQP if time remains
    if time_left() > 15:
        for _ in range(8):
            if time_left() < 8:
                break
            try:
                pts = np.zeros((N, 2))
                for i in range(N):
                    u = rng.random()
                    v = rng.random()
                    while u + v > 1.0:
                        u = rng.random()
                        v = rng.random()
                    pts[i, 0] = u
                    pts[i, 1] = v
                init_bary = _project_to_bary_simplex(pts)
                
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=100)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = bary_slsqp
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
    
    # Final high-precision polish on the best solution
    if time_left() > 12:
        try:
            bounds_final = [(0, 1)] * (2 * N)
            polished = _final_polish(best_x.flatten(), bounds_final, time_budget=min(35.0, time_left() - 8))
            polished_bary = _project_to_bary_simplex(polished.reshape(N, 2))
            polished_score = _min_area_ratio_from_bary(polished_bary)
            if polished_score > best_score:
                best_x = polished_bary
                best_score = polished_score
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
