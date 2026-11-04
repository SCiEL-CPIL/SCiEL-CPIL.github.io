"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute indices for all C(11,3) = 165 triples
_IDX_I, _IDX_J, _IDX_K = None, None, None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _param_to_points(params):
    """
    Map parameters s, t in [0, 1] to points inside the triangle.
    params: shape (2N,)
    Returns: shape (N, 2)
    """
    s = params[0::2]
    t = params[1::2]
    # Clip to ensure strictly inside for numerical stability, though [0,1] is valid
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    t = np.clip(t, 1e-9, 1.0 - 1e-9)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y = s * t * H  =>  s * t = y / H
    # x = s - 0.5 * s * t = s - 0.5 * (y/H)
    # s = x + 0.5 * y / H
    s = x + 0.5 * y / H
    st = y / H
    
    # t = st / s
    t = np.zeros_like(s)
    mask = s > 1e-9
    t[mask] = st[mask] / s[mask]
    
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    t = np.clip(t, 1e-9, 1.0 - 1e-9)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Cross product: (pj - pi) x (pk - pi)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _objective_hard(params):
    """Objective for DE: negative min triangle area (non-smooth)."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _objective_soft(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    # soft_min = min_a - (1/beta) * log(sum(exp(-beta * (areas - min_a))))
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _gradient_soft(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Weights for soft-min: w_k = exp(-beta * (a_k - min_a)) / sum(exp(-beta * (a_j - min_a)))
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(soft_min)/d(area_k) = -w_k
    # We want gradient of -soft_min, so d(-soft_min)/d(area_k) = w_k
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
    # area = 0.5 * |cross|
    # d(area)/d(cross) = 0.5 * sign(cross)
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    # d(cross)/d(pi), d(cross)/d(pj), d(cross)/d(pk)
    # cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    
    # d(cross)/d(pi_x) = -(pk_y - pi_y) + (pj_y - pi_y)
    # d(cross)/d(pi_y) = -(pj_x - pi_x) + (pk_x - pi_x)
    # d(cross)/d(pj_x) = (pk_y - pi_y)
    # d(cross)/d(pj_y) = -(pk_x - pi_x)
    # d(cross)/d(pk_x) = -(pj_y - pi_y)
    # d(cross)/d(pk_y) = (pj_x - pi_x)
    
    # Gradient of area wrt points
    # grad_area_pi = 0.5 * sign(cross) * [-(pk_y - pi_y) + (pj_y - pi_y), -(pj_x - pi_x) + (pk_x - pi_x)]
    # grad_area_pj = 0.5 * sign(cross) * [pk_y - pi_y, -(pk_x - pi_x)]
    # grad_area_pk = 0.5 * sign(cross) * [-(pj_y - pi_y), pj_x - pi_x]
    
    # We need gradient of -soft_min wrt points, which is sum_k w_k * grad_area_k
    # So we accumulate w_k * grad_area_k for each point
    
    grad_points = np.zeros((N, 2))
    
    # Coefficients
    coeff = 0.5 * weights * sign_cross
    
    # Contribution from pi
    d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    np.add.at(grad_points, i_idx, coeff[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    
    # Contribution from pj
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    np.add.at(grad_points, j_idx, coeff[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    
    # Contribution from pk
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    np.add.at(grad_points, k_idx, coeff[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    # Now chain rule: d(points)/d(params)
    # points[i] = [s - 0.5 st, st H]
    # x = s - 0.5 s t
    # y = s t H
    
    # dx/ds = 1 - 0.5 t
    # dx/dt = -0.5 s
    # dy/ds = t H
    # dy/dt = s H
    
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    t = np.clip(t, 1e-9, 1.0 - 1e-9)
    
    dx_ds = 1.0 - 0.5 * t
    dx_dt = -0.5 * s
    dy_ds = t * H
    dy_dt = s * H
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    # Objective is -soft_min, gradient is -d(soft_min)/d(params)
    # Wait, my grad_points is d(soft_min)/d(points) because I summed w_k * grad_area_k.
    # So d(soft_min)/d(params) = grad_points_chain.
    # We want to minimize -soft_min, so gradient is -grad_points_chain.
    return -grad_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling in barycentric space.
    """
    params = np.zeros(2 * n_points)
    # Start at centroid: s=1/3? No, centroid in barycentric (1/3, 1/3, 1/3) means s=2/3?
    # lambda1 = s(1-t), lambda2 = st, lambda3 = 1-s
    # Centroid: lambda1=1/3, lambda2=1/3, lambda3=1/3
    # 1-s = 1/3 => s = 2/3
    # st = 1/3 => (2/3)t = 1/3 => t = 1/2
    params[0] = 2/3
    params[1] = 1/2
    
    points = _param_to_points(params)
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        # Map to triangle in barycentric (s,t) space?
        # Uniform sampling in triangle is easier in Cartesian, then convert.
        # Or just sample s,t uniformly in [0,1] and accept if inside?
        # The mapping (s,t) -> Triangle is a bijection from [0,1]^2 to the triangle.
        # So uniform s,t is NOT uniform in area.
        # Better to sample Cartesian uniformly and convert to params.
        
        # Sample Cartesian uniformly
        # Rejection sampling or transformation
        u = rng.uniform(0, 1, n_candidates)
        v = rng.uniform(0, 1, n_candidates)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        cx = u + 0.5 * v
        cy = H * v
        candidates = np.column_stack([cx, cy])
        
        # Convert candidates to params
        cand_params = _points_to_param(candidates)
        
        # Compute distances from existing points to candidates
        # Existing points
        pts_arr = points
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        best_idx = np.argmax(min_dist_sq)
        
        # Add best candidate
        params[2*k] = cand_params[best_idx, 0]
        params[2*k+1] = cand_params[best_idx, 1]
        points[k] = candidates[best_idx]
        
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initial candidates using FPS
    inits = []
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        if time.time() > deadline - 60:
            break
        r = np.random.RandomState(seed)
        p = _generate_fps_initialization(r, n_points=N, n_candidates=300)
        inits.append(p)
        
        # Evaluate
        score = -_objective_hard(p)
        if score > best_score:
            best_score = score
            best_params = p.copy()
            
    # Phase 2: Differential Evolution (Global Search)
    # Use hard objective for DE
    bounds = [(1e-9, 1.0 - 1e-9)] * (2 * N)
    
    if time.time() < deadline - 50:
        try:
            de_result = differential_evolution(
                _objective_hard,
                bounds,
                maxiter=50,
                popsize=20,
                tol=0.001,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            score = -de_result.fun
            if score > best_score:
                best_score = score
                best_params = de_result.x.copy()
                
        except Exception:
            pass
            
    # Phase 3: Local Refinement with L-BFGS-B using Soft Min
    # We will run L-BFGS-B on the best candidate and several other good inits
    # Using a high beta for soft min makes it closer to hard min but smooth
    
    candidates_to_refine = [best_params]
    # Add top few inits
    scored_inits = [( -_objective_hard(p), p) for p in inits]
    scored_inits.sort(reverse=True, key=lambda x: x[0])
    for s, p in scored_inits[:5]:
        candidates_to_refine.append(p)
        
    # Add perturbations of best
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        perturbed = best_params + rng.randn(2*N) * 0.05
        perturbed = np.clip(perturbed, 1e-9, 1.0 - 1e-9)
        candidates_to_refine.append(perturbed)
        
    beta = 200.0  # High beta for sharp min
    
    for i, init_p in enumerate(candidates_to_refine):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective_soft,
                init_p,
                jac=lambda x: _gradient_soft(x, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Evaluate with hard objective to get true score
            score = -_objective_hard(res.x)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    # Phase 4: Simulated Annealing on Hard Objective + L-BFGS-B Polish
    if best_params is not None and time.time() < deadline - 15:
        try:
            current = best_params.copy()
            current_score = -_objective_hard(current)
            best_sa = current.copy()
            best_sa_score = current_score
            
            t0 = 0.05
            t_end = 0.001
            n_iter = 2000
            cooling = (t_end / t0) ** (1.0 / n_iter)
            t = t0
            
            for i in range(n_iter):
                if time.time() > deadline - 5:
                    break
                t *= cooling
                scale = 0.1 * t / t0
                perturbed = current + rng.randn(2*N) * scale
                perturbed = np.clip(perturbed, 1e-9, 1.0 - 1e-9)
                
                perturbed_score = -_objective_hard(perturbed)
                delta = perturbed_score - current_score
                
                if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
                    current = perturbed
                    current_score = perturbed_score
                    
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa = current.copy()
                        
            if best_sa_score > best_score:
                best_score = best_sa_score
                best_params = best_sa.copy()
                
            # Final Polish with L-BFGS-B
            if time.time() < deadline - 2:
                res = minimize(
                    _objective_soft,
                    best_params,
                    jac=lambda x: _gradient_soft(x, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                score = -_objective_hard(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception:
            pass
            
    # Fallback
    if best_params is None:
        best_params = _generate_fps_initialization(rng, n_points=N)
        
    # Convert best params to points
    best_points = _param_to_points(best_params)
    
    # Final check and clip (just in case)
    x = best_points[:, 0]
    y = best_points[:, 1]
    # Ensure inside
    # y >= 0
    # y <= sqrt(3) x  => x >= y/sqrt(3)
    # y <= sqrt(3) (1-x) => x <= 1 - y/sqrt(3)
    
    # If any point is slightly outside due to floating point, clip it
    # But our param space [0,1]x[0,1] should guarantee inside.
    
    return best_points
