import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _smoothed_min_area(points, beta=100.0):
    """Compute the smoothed minimum area using Log-Sum-Exp."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Smooth min: (1/beta) * log(sum(exp(-beta * areas)))
    # Subtract max for numerical stability
    max_a = np.max(areas)
    # If max_a is very small, exp(-beta*(areas-max_a)) might be exp(0)=1
    # If max_a is large, it's fine.
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    return log_sum_exp / beta

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = bary.copy()
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_de(x):
    """Objective for Differential Evolution."""
    bary = x.reshape(-1, 2)
    s = bary[:, 0] + bary[:, 1]
    if np.any(s > 1.0):
        return 1000.0 # Penalty for invalid points
    points = _bary_to_cart(bary)
    return -_min_triangle_area(points)

def _objective_slsqp(x, beta):
    """Objective for SLSQP with smoothed min."""
    bary = x.reshape(-1, 2)
    # We assume SLSQP keeps points within bounds/constraints, 
    # but we clamp just in case for the area calculation
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return _smoothed_min_area(points, beta)

def _make_slsqp_constraints():
    """Create inequality constraints for SLSQP: u>=0, v>=0, u+v<=1."""
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    return constraints

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    bounds = [(0.0, 1.0)] * (2 * N)
    constraints = _make_slsqp_constraints()
    
    best_points = None
    best_score = -np.inf
    best_bary = None

    # 1. Global Search with Differential Evolution
    if time_left() > 30:
        try:
            de_result = differential_evolution(
                _objective_de, bounds, 
                maxiter=50, popsize=15, 
                seed=42, polish=False, tol=0.01
            )
            if de_result.fun < 1000:
                best_bary = _clamp_bary(de_result.x.reshape(-1, 2))
                best_points = _bary_to_cart(best_bary)
                best_score = _min_triangle_area(best_points)
        except Exception:
            pass

    # 2. Local Refinement with SLSQP and Smoothed Objective
    # We will refine the best candidate from DE, and also some random starts
    
    candidates = []
    if best_bary is not None:
        candidates.append(best_bary)
        
    # Add some random valid starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        bary_rand = []
        for _ in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                # Reflect or scale
                # Scale to fit
                s = u + v
                u = u / s
                v = v / s
            bary_rand.append([u, v])
        candidates.append(_clamp_bary(np.array(bary_rand)))

    # Refine each candidate
    betas = [50.0, 100.0, 200.0]
    
    for bary_start in candidates:
        if time_left() < 10:
            break
            
        current_bary = bary_start.copy()
        current_score = _min_triangle_area(_bary_to_cart(current_bary))
        
        for beta in betas:
            if time_left() < 5:
                break
            try:
                res = minimize(
                    _objective_slsqp, 
                    current_bary.flatten(), 
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if res.success:
                    new_bary = _clamp_bary(res.x.reshape(-1, 2))
                    new_points = _bary_to_cart(new_bary)
                    new_score = _min_triangle_area(new_points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        current_bary = new_bary.copy()
                        
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = _bary_to_cart(current_bary).copy()
            best_bary = current_bary.copy()

    # Fallback
    if best_points is None:
        bary_fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75]
        ])
        best_points = _bary_to_cart(_clamp_bary(bary_fallback))
        best_score = _min_triangle_area(best_points)
        best_bary = _clamp_bary(bary_fallback)

    return best_points
