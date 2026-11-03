"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Combines barycentric coordinate representation with smooth min-surrogate optimization,
FPS initializations, and multi-stage refinement (SLSQP + Nelder-Mead)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _bary_to_cart(bc):
    """Convert barycentric (b, c) to Cartesian. a = 1 - b - c.
    x = b + 0.5*c, y = c*sqrt(3)/2
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])

def _clip_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    y = np.clip(y, 0, np.sqrt(3) / 2)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])

def _min_triangle_area_cart(points):
    """Compute min area among all triangles, vectorized in Cartesian coords."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _dets_from_bary(bc):
    """Compute all 165 determinants (proportional to area) from barycentric coords."""
    i_idx, j_idx, k_idx = _get_indices()
    u = bc[:, 0]
    v = bc[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """Smooth objective using soft-min for L-BFGS-B. x is (2N,) barycentric (b,c) pairs."""
    bc = x.reshape(N, 2)
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    
    # Penalty for invalid barycentric
    penalty = 0.0
    excess = np.maximum(0.0, -a) + np.maximum(0.0, -b) + np.maximum(0.0, -c)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _dets_from_bary(bc)
    areas = np.abs(dets)
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta)*log(sum(exp(-beta*(areas-m))))
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """Objective for SLSQP: minimize -min_area. x is (2N,) barycentric (b,c) pairs."""
    bc = x.reshape(N, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1)
    a = 1.0 - b - c
    # If invalid, return large penalty
    if np.any(a < -1e-9) or np.any(b < -1e-9) or np.any(c < -1e-9):
        return 1.0 + penalty_lambda * np.sum(np.maximum(0, -a)**2 + np.maximum(0, -b)**2 + np.maximum(0, -c)**2)
    dets = _dets_from_bary(bc)
    return -np.min(np.abs(dets))

def _generate_fps_initialization(rng, n_points=N, n_candidates=400):
    """Generate initial config using Farthest Point Sampling in barycentric space."""
    # Start with centroid in barycentric: a=b=c=1/3 -> b=1/3, c=1/3
    bc = np.zeros((n_points, 2))
    bc[0] = [1.0/3, 1.0/3]
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([r1, r2])
        
        # Compute Cartesian distances
        cand_cart = _bary_to_cart(candidates)
        exist_cart = _bary_to_cart(bc[:k])
        diffs = cand_cart[:, np.newaxis, :] - exist_cart[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        bc[k] = candidates[best_idx]
    
    return bc

def _cart_to_bary(points):
    """Convert Cartesian to barycentric (b, c)."""
    x = points[:, 0]
    y = points[:, 1]
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    return np.column_stack([b, c])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_bc = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Bounds and constraints for barycentric SLSQP
    # b >= 0, c >= 0, a = 1-b-c >= 0
    bounds = [(0.0, 1.0)] * (2 * N)
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    # 1. Generate diverse initializations
    initial_bcs = []
    n_fps = 8
    for i in range(n_fps):
        if time_left() < 15.0:
            break
        bc = _generate_fps_initialization(rng)
        initial_bcs.append(bc.flatten())
    
    # Add random initializations
    for i in range(5):
        if time_left() < 15.0:
            break
        bc = np.zeros((N, 2))
        for k in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            bc[k] = [u, v]
        initial_bcs.append(bc.flatten())
    
    # 2. Optimize each initialization
    for bc0 in initial_bcs:
        if time_left() < 8.0:
            break
        
        # Stage 1: SLSQP
        try:
            res = minimize(
                _objective_slsqp, bc0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            bc_opt = np.clip(res.x, 0, 1)
            # Ensure validity
            a = 1.0 - bc_opt[0::2] - bc_opt[1::2]
            if np.any(a < -1e-6):
                # Clip to valid region
                for i in range(N):
                    if bc_opt[2*i] + bc_opt[2*i+1] > 1.0:
                        s = bc_opt[2*i] + bc_opt[2*i+1]
                        bc_opt[2*i] /= s
                        bc_opt[2*i+1] /= s
            score = _min_triangle_area_cart(_bary_to_cart(bc_opt.reshape(N, 2)))
            if score > best_score:
                best_score = score
                best_bc = bc_opt.copy()
        except Exception:
            pass
        
        # Stage 2: Nelder-Mead polish
        if time_left() < 10.0:
            break
        try:
            res2 = minimize(
                _objective_slsqp, bc0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            bc_opt2 = np.clip(res2.x, 0, 1)
            a2 = 1.0 - bc_opt2[0::2] - bc_opt2[1::2]
            if np.any(a2 < -1e-6):
                for i in range(N):
                    if bc_opt2[2*i] + bc_opt2[2*i+1] > 1.0:
                        s = bc_opt2[2*i] + bc_opt2[2*i+1]
                        bc_opt2[2*i] /= s
                        bc_opt2[2*i+1] /= s
            score2 = _min_triangle_area_cart(_bary_to_cart(bc_opt2.reshape(N, 2)))
            if score2 > best_score:
                best_score = score2
                best_bc = bc_opt2.copy()
        except Exception:
            pass
    
    # 3. If time permits, do additional perturbation restarts from best
    if best_bc is not None and time_left() > 20.0:
        for i in range(4):
            if time_left() < 10.0:
                break
            try:
                noise = rng.normal(0, 0.005, size=2*N)
                perturbed = np.clip(best_bc + noise, 0, 1)
                # Ensure validity
                for idx in range(N):
                    if perturbed[2*idx] + perturbed[2*idx+1] > 1.0:
                        s = perturbed[2*idx] + perturbed[2*idx+1]
                        perturbed[2*idx] /= s
                        perturbed[2*idx+1] /= s
                
                res = minimize(
                    _objective_slsqp, perturbed, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                bc_opt = np.clip(res.x, 0, 1)
                a = 1.0 - bc_opt[0::2] - bc_opt[1::2]
                if np.any(a < -1e-6):
                    for idx in range(N):
                        if bc_opt[2*idx] + bc_opt[2*idx+1] > 1.0:
                            s = bc_opt[2*idx] + bc_opt[2*idx+1]
                            bc_opt[2*idx] /= s
                            bc_opt[2*idx+1] /= s
                score = _min_triangle_area_cart(_bary_to_cart(bc_opt.reshape(N, 2)))
                if score > best_score:
                    best_score = score
                    best_bc = bc_opt.copy()
            except Exception:
                pass
    
    # 4. Final Nelder-Mead polish if time allows
    if best_bc is not None and time_left() > 15.0:
        try:
            res = minimize(
                _objective_slsqp, best_bc, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            bc_opt = np.clip(res.x, 0, 1)
            a = 1.0 - bc_opt[0::2] - bc_opt[1::2]
            if np.any(a < -1e-6):
                for idx in range(N):
                    if bc_opt[2*idx] + bc_opt[2*idx+1] > 1.0:
                        s = bc_opt[2*idx] + bc_opt[2*idx+1]
                        bc_opt[2*idx] /= s
                        bc_opt[2*idx+1] /= s
            score = _min_triangle_area_cart(_bary_to_cart(bc_opt.reshape(N, 2)))
            if score > best_score:
                best_score = score
                best_bc = bc_opt.copy()
        except Exception:
            pass
    
    if best_bc is None:
        # Fallback: centered grid
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return _clip_to_triangle(fallback_points)
    
    points = _bary_to_cart(best_bc.reshape(N, 2))
    points = _clip_to_triangle(points)
    return np.array(points)
