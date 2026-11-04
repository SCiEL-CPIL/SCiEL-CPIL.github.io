# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
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

def _simulated_annealing(bary0, time_budget=60.0, T_init=0.01, T_min=0.0001, cooling=0.97):
    """
    Simulated annealing with point swaps and local jitter.
    """
    start = time.time()
    current = bary0.copy()
    current_score = _min_area_ratio_from_bary(current)
    best = current.copy()
    best_score = current_score
    no_improve_count = 0
    max_no_improve = 200
    T = T_init
    
    while time.time() - start < time_budget:
        move_type = np.random.random()
        
        if move_type < 0.3:
            i, j = np.random.choice(N, size=2, replace=False)
            candidate = current.copy()
            candidate[i], candidate[j] = candidate[j].copy(), candidate[i].copy()
            candidate = _project_to_bary_simplex(candidate)
        elif move_type < 0.6:
            i = np.random.randint(N)
            candidate = current.copy()
            sigma = T * 5
            candidate[i] += np.random.normal(0, sigma, size=2)
            candidate = _project_to_bary_simplex(candidate)
        else:
            candidate = current.copy()
            sigma = T * 2
            candidate += np.random.normal(0, sigma, size=(N, 2))
            candidate = _project_to_bary_simplex(candidate)
        
        cand_score = _min_area_ratio_from_bary(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = current_score - cand_score
            if delta < T:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current = candidate
                    current_score = cand_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        if no_improve_count > max_no_improve:
            T = max(T, current_score * 0.02)
            no_improve_count = 0
    
    return best, best_score

def _solve_lp_for_orientation(bary_init):
    """
    Solve the LP for a fixed orientation pattern (sign of each determinant).
    Variables: [u_0, v_0, u_1, v_1, ..., u_10, v_10, t]
    Maximize t subject to:
      - For each triple with positive orientation: det >= t
      - For each triple with negative orientation: det <= -t
      - u_i >= 0, v_i >= 0, u_i + v_i <= 1
    """
    # Determine orientation from initial barycentric coords
    dets_init = _all_dets_from_bary(bary_init)
    orientation = np.sign(dets_init)
    # Treat zero as positive
    orientation[orientation == 0] = 1
    
    # Number of variables: 2*N + 1 (for t)
    n_vars = 2 * N + 1
    
    # Objective: minimize -t (to maximize t)
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -t
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For positive orientation: det >= t  =>  -det <= -t
    # For negative orientation: det <= -t  =>  -det >= t  =>  det <= -t
    
    A_rows = []
    b_rows = []
    
    for m in range(N_TRIPLES):
        i, j, k = _IDX_TRIPLES[m]
        # det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
        # where w = 1 - u - v
        # This is linear in u_i, v_i, u_j, v_j, u_k, v_k
        
        # Coefficients for u_i, v_i, u_j, v_j, u_k, v_k
        # Let's expand det in terms of u and v only
        # w_i = 1 - u_i - v_i, etc.
        
        # det = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k) 
        #       - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #       + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        
        # This is getting complex. Let me compute it differently.
        # The determinant of [u_i, v_i, w_i; u_j, v_j, w_j; u_k, v_k, w_k]
        # where w = 1-u-v
        
        # Let me just compute the linear coefficients directly.
        # det = u_i*(v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k)
        #       - v_i*(u_j - u_j*u_k - u_j*v_k - u_k + u_j*u_k + u_j*v_k)
        #       + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        
        # This is still messy. Let me use a different approach.
        # I'll compute the determinant as a function of the 6 variables and extract coefficients.
        
        # Actually, let me just build the constraint matrix numerically.
        # For each variable, compute the partial derivative of det.
        
        # det is linear in each of u_i, v_i, u_j, v_j, u_k, v_k
        # (because w = 1-u-v, so each term is linear)
        
        # Let me compute by evaluating det at basis points.
        # Actually, it's easier to just expand algebraically.
        
        # w_i = 1 - u_i - v_i
        # w_j = 1 - u_j - v_j  
        # w_k = 1 - u_k - v_k
        
        # det = u_i * (v_j * w_k - w_j * v_k) - v_i * (u_j * w_k - w_j * u_k) + w_i * (u_j * v_k - v_j * u_k)
        
        # Let me expand each term:
        # Term1 = u_i * (v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k)
        #        = u_i * (v_j - v_j*u_k - v_j*v_k - v_k + u_j*v_k + v_j*v_k)
        #        = u_i * (v_j - v_j*u_k - v_k + u_j*v_k)
        #        = u_i*v_j - u_i*v_j*u_k - u_i*v_k + u_i*u_j*v_k
        
        # Term2 = -v_i * (u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
        #        = -v_i * (u_j - u_j*u_k - u_j*v_k - u_k + u_j*u_k + u_j*v_k)
        #        = -v_i * (u_j - u_k)
        #        = -v_i*u_j + v_i*u_k
        
        # Term3 = (1-u_i-v_i) * (u_j*v_k - v_j*u_k)
        #        = u_j*v_k - v_j*u_k - u_i*u_j*v_k + u_i*v_j*u_k - v_i*u_j*v_k + v_i*v_j*u_k
        
        # Sum:
        # det = u_i*v_j - u_i*v_j*u_k - u_i*v_k + u_i*u_j*v_k
        #       - v_i*u_j + v_i*u_k
        #       + u_j*v_k - v_j*u_k - u_i*u_j*v_k + u_i*v_j*u_k - v_i*u_j*v_k + v_i*v_j*u_k
        
        # Cancel terms:
        # -u_i*v_j*u_k + u_i*v_j*u_k = 0
        # u_i*u_j*v_k - u_i*u_j*v_k = 0
        
        # det = u_i*v_j - u_i*v_k - v_i*u_j + v_i*u_k + u_j*v_k - v_j*u_k - v_i*u_j*v_k + v_i*v_j*u_k
        
        # Wait, I think I made an error. Let me redo this more carefully.
        
        # Actually, the determinant of a 3x3 matrix with rows [u_i, v_i, 1-u_i-v_i] is:
        # u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k) - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k) + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
        
        # This is bilinear in the variables, not linear! Because we have products like u_i*v_j.
        
        # Oh no, the determinant is NOT linear in the barycentric coordinates. It's quadratic.
        # So the LP approach doesn't directly work.
        
        # Let me reconsider. The area of a triangle formed by three points in barycentric coordinates
        # is proportional to the absolute value of the determinant, which is a multilinear function.
        
        # So the LP approach as stated doesn't work directly. We need a different approach.
        
        # Let me fall back to using the current best orientation and just do local optimization.
        pass
    
    # Since the determinant is not linear, we can't use a simple LP.
    # Return the input unchanged.
    return bary_init.copy()

def _final_polish(x0, bounds, time_budget=30.0):
    """
    Final high-precision polish: SLSQP with very tight tolerances,
    then a second SLSQP pass, then Nelder-Mead with adaptive.
    """
    start = time.time()
    current = x0.copy()
    best_score = _min_area_ratio_from_bary(current.reshape(N, 2))
    
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
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    inits.append(_structured_init_fps().flatten())
    
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
        
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
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
    
    for init in inits:
        if time_left() < 5:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            best_local_x = init_bary
            best_local_score = _min_area_ratio_from_bary(init_bary)
            
            if time_left() > 10:
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=200)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                if score_slsqp > best_local_score:
                    best_local_score = score_slsqp
                    best_local_x = bary_slsqp
                    init_bary = bary_slsqp

            if time_left() > 10:
                x_nm = _local_optimize_nm(init_bary.flatten(), bounds, maxiter=300)
                bary_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score_nm = _min_area_ratio_from_bary(bary_nm)
                if score_nm > best_local_score:
                    best_local_score = score_nm
                    best_local_x = bary_nm
                    init_bary = bary_nm

            if time_left() > 15:
                sa_bary, sa_score = _simulated_annealing(init_bary, time_budget=min(30.0, time_left() - 10))
                if sa_score > best_local_score:
                    best_local_score = sa_score
                    best_local_x = sa_bary
                    init_bary = sa_bary

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    if time_left() > 20:
        for _ in range(10):
            if time_left() < 10:
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
    
    if time_left() > 15:
        try:
            bounds_final = [(0, 1)] * (2 * N)
            polished = _final_polish(best_x.flatten(), bounds_final, time_budget=min(40.0, time_left() - 10))
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
