"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_exact_score(points):
    """
    Compute the exact packing radius for a given set of points.
    Returns the minimum of:
    1. Half the minimum pairwise distance.
    2. The minimum distance to the boundary (1 - |p_i|).
    """
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Boundary check
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    # Pairwise distances
    # Use upper triangle to avoid redundancy and self-distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = points[i_idx] - points[j_idx]
    pair_dists = np.linalg.norm(diffs, axis=1)
    min_pair = np.min(pair_dists) / 2.0
    
    return min(min_pair, min_boundary)

def _hex_initialization(n):
    """
    Generate a hexagonal lattice initialization clipped to the unit circle.
    We try a few spacing factors to ensure we get at least n points,
    then select the n points closest to the center (or a mix to ensure boundary coverage).
    """
    best_pts = None
    best_score = -1.0
    
    # Theoretical max radius is ~ 1/sqrt(N) for dense packing, but hex is ~ 1/sqrt(N*sqrt(3)/2) approx?
    # Area of hex cell with side s is s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 <= pi  => s <= sqrt(2*pi / (N*sqrt(3))).
    # For N=300, s ~ sqrt(6.28 / 519) ~ sqrt(0.012) ~ 0.11.
    # Diameter = 2s ~ 0.22. r ~ 0.11? No, r=s.
    # Let's just sweep a range of spacings.
    
    for scale in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        s = (2.0 / np.sqrt(n)) * scale
        # Hex grid generation
        # Rows are spaced s * sqrt(3) / 2
        row_height = s * np.sqrt(3.0) / 2.0
        
        # Number of rows to cover radius 1
        max_row = int(1.0 / row_height) + 2
        max_col = int(1.0 / s) + 2
        
        pts_list = []
        for j in range(-max_row, max_row + 1):
            y = j * row_height
            # Offset even/odd rows
            offset = 0.0 if j % 2 == 0 else s / 2.0
            for i in range(-max_col, max_col + 1):
                x = i * s + offset
                r = np.hypot(x, y)
                # Keep points strictly inside, with a small margin to ensure they can be pushed out
                if r < 1.0:
                    pts_list.append([x, y])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        
        # Strategy: Take points with smallest norms first?
        # Actually, for a disk, taking the most central points creates a solid core,
        # but we need to fill the boundary too.
        # A better heuristic: Sort by norm, but ensure we don't leave large holes.
        # For initialization, taking the n points with smallest norms is standard and safe.
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        score = _compute_exact_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    if best_pts is None:
        # Fallback: Random
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
        best_score = _compute_exact_score(best_pts)
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points, best_score = _hex_initialization(N)
    best_points = points.copy()
    
    # 2. Optimization using Soft-Min objective
    # Objective: minimize -soft_min(v_i) where v_i are the "slack" variables (distances).
    # v_i = d_ij / 2 for pairs, v_k = 1 - |p_k| for boundary.
    # We want to maximize min(v_i).
    # soft_min(v) = -1/beta * log(sum(exp(-beta * v_i)))
    # grad(soft_min) = -1/sum(exp(-beta*v)) * sum(exp(-beta*v) * (-beta * grad(v)))
    #                = sum(exp(-beta*v) * grad(v)) / sum(exp(-beta*v))
    #                = weighted average of gradients of active constraints.
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def soft_min_objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Check for non-finite or out of bounds
        if not np.all(np.isfinite(p)):
            return 1e10, np.zeros_like(p)
        
        # Compute pairwise distances
        diffs = p[i_idx] - p[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero in gradient
        dists_safe = np.maximum(dists, 1e-12)
        
        # Pair slacks: d/2
        v_pairs = dists / 2.0
        
        # Boundary slacks: 1 - |p|
        norms = np.linalg.norm(p, axis=1)
        v_bounds = 1.0 - norms
        
        # Combine all slacks
        # We treat them as a single list for the log-sum-exp
        # To avoid memory issues with huge exp, we can use the log-sum-exp trick
        # log(sum(exp(-beta * v))) = -beta * min(v) + log(sum(exp(-beta * (v - min(v)))))
        
        all_v = np.concatenate([v_pairs, v_bounds])
        
        # Numerical stability
        min_v = np.min(all_v)
        shifted_v = all_v - min_v
        exp_terms = np.exp(-beta * shifted_v)
        sum_exp = np.sum(exp_terms)
        
        # Objective value: -soft_min = -(-1/beta * log(sum_exp)) + min_v?
        # soft_min = min_v - 1/beta * log(sum_exp)
        # We want to minimize -soft_min = 1/beta * log(sum_exp) - min_v
        # Wait, standard log-sum-exp for min:
        # min(x) approx -1/beta log sum exp(-beta x)
        # So we minimize - ( -1/beta log sum exp(-beta v) ) = 1/beta log sum exp(-beta v)
        # Let's stick to: f = - soft_min(v)
        # soft_min(v) = min_v - (1/beta) * log( sum(exp(-beta * (v_i - min_v))) )
        # f = -min_v + (1/beta) * log(sum_exp)
        
        obj_val = -min_v + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # grad(f) = - grad(soft_min)
        # grad(soft_min) = (1/sum_exp) * sum(exp(-beta * v_i) * grad(v_i))
        # Note: The shift min_v is constant w.r.t x, so it doesn't affect gradient.
        # The weights are w_i = exp(-beta * (v_i - min_v)) / sum_exp
        
        weights = exp_terms / sum_exp
        
        grad = np.zeros_like(p)
        
        # Gradient of v_pairs (d/2)
        # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij)
        # d/dp_j (d_ij/2) = (p_j - p_i) / (2 * d_ij)
        # Vectorized update:
        # grad_pairs = 0.5 * (p[i] - p[j]) / d_ij
        
        # We need to accumulate into grad array.
        # Using np.add.at for efficiency
        vecs = 0.5 * diffs / dists_safe[:, np.newaxis]
        
        # Contribution to i
        np.add.at(grad, i_idx, weights[:num_pairs][:, np.newaxis] * vecs)
        # Contribution to j (negative)
        np.add.at(grad, j_idx, -weights[:num_pairs][:, np.newaxis] * vecs)
        
        # Gradient of v_bounds (1 - |p|)
        # d/dp_i (1 - |p_i|) = - p_i / |p_i|
        # If p_i is 0, gradient is 0 (or undefined, but we can ignore as it's measure 0)
        norms_safe = np.maximum(norms, 1e-12)
        grad_bounds = -p / norms_safe[:, np.newaxis]
        
        # Weighted sum for boundary
        w_bounds = weights[num_pairs:]
        grad += w_bounds[:, np.newaxis] * grad_bounds
        
        # Final gradient of f = -soft_min is - grad(soft_min)
        # Wait.
        # f = - soft_min(v)
        # grad(f) = - grad(soft_min(v))
        # We calculated grad(soft_min) as the weighted average of grad(v).
        # So we return -grad.
        
        return obj_val, -grad

    # Multi-stage optimization
    # Start with low beta (smooth) to escape shallow local minima, increase beta for precision.
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_pts = best_points
    current_score = best_score
    
    for be in betas:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                lambda x: soft_min_objective_and_grad(x, be)[0],
                current_pts.flatten(),
                jac=lambda x: soft_min_objective_and_grad(x, be)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_exact_score(candidate)
            
            if candidate_score > current_score:
                current_pts = candidate
                current_score = candidate_score
                best_points = current_pts.copy()
                best_score = current_score
                
        except Exception as e:
            # If optimization fails, continue with next beta
            pass
            
    # Final polish: Use very high beta and tight tolerance
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda x: soft_min_objective_and_grad(x, 2000.0)[0],
                best_points.flatten(),
                jac=lambda x: soft_min_objective_and_grad(x, 2000.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 50,
                    'disp': False
                }
            )
            candidate_final = res_final.x.reshape(-1, 2)
            final_score = _compute_exact_score(candidate_final)
            
            if final_score > best_score:
                best_points = candidate_final
                best_score = final_score
        except Exception:
            pass

    # Final validation and cleanup
    # Ensure points are strictly inside. If any point is outside, scale it in slightly.
    # This is a safety net, shouldn't happen if optimizer works correctly.
    norms = np.linalg.norm(best_points, axis=1)
    outside_mask = norms > 1.0
    if np.any(outside_mask):
        # Scale points that are outside back to just inside
        # This might violate packing, but it's better than invalid.
        # Actually, if a point is outside, the score is 0 (or negative).
        # We should try to fix it by moving it in.
        for i in np.where(outside_mask)[0]:
            best_points[i] = best_points[i] * (0.999 / norms[i])
            
    # Recalculate score after any adjustments
    final_score = _compute_exact_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
