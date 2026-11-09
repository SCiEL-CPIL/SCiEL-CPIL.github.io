import numpy as np
import time
from scipy.spatial.distance import pdist
from itertools import combinations
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _get_candidate_pool() -> np.ndarray:
    """
    Generate a pool of high-symmetry candidate points on the unit sphere.
    Combines vertices of Icosahedron (12), Dodecahedron (20), and Octahedron (6).
    Total unique points will be less than 38 due to overlaps, but we keep them
    distinct enough for the selection process.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Dodecahedron vertices (20)
    dodeca_verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dodeca_verts.append([sx, sy, sz])
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            dodeca_verts.append([0, sx/phi, sz*phi])
            dodeca_verts.append([sx/phi, sz*phi, 0])
            dodeca_verts.append([sz*phi, 0, sx/phi])
            
    dodeca_verts = np.array(dodeca_verts, dtype=float)
    dodeca_verts /= np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    
    # Octahedron vertices (6)
    octa_verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    octa_verts /= np.linalg.norm(octa_verts, axis=1, keepdims=True)
    
    # Combine and deduplicate
    all_verts = np.vstack([ico_verts, dodeca_verts, octa_verts])
    
    # Simple deduplication based on rounding to handle floating point noise
    # We want to keep distinct geometric positions.
    unique_verts = []
    for v in all_verts:
        rounded = np.round(v, 4)
        is_dup = False
        for u in unique_verts:
            if np.allclose(u, rounded, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            unique_verts.append(v)
            
    return np.array(unique_verts)

def _refine_points(points: np.ndarray, start_time: float, deadline: float) -> np.ndarray:
    """
    Locally optimize the selected points using L-BFGS-B with a smooth metric.
    This step corrects any slight deviations from the optimal local configuration
    induced by the discrete selection.
    """
    if time.time() > deadline - 10:
        return points
        
    def smooth_metric(params, beta=50.0):
        pts = params.reshape(14, 3)
        # Normalize to unit sphere to remove scale
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        dists = pdist(pts)
        if np.any(~np.isfinite(dists)):
            return 1e10
            
        # Soft min/max
        min_d = np.min(dists)
        max_d = np.max(dists)
        
        # Log-sum-exp approximation
        beta_val = beta
        shifted_min = -beta_val * (dists - min_d)
        shifted_min = np.clip(shifted_min, -700, 700)
        sum_exp_min = np.sum(np.exp(shifted_min))
        log_sum_exp_min = np.log(sum_exp_min) if sum_exp_min > 0 else 0.0
        d_min_soft = min_d - (log_sum_exp_min / beta_val)
        
        shifted_max = beta_val * (dists - max_d)
        shifted_max = np.clip(shifted_max, -700, 700)
        sum_exp_max = np.sum(np.exp(shifted_max))
        log_sum_exp_max = np.log(sum_exp_max) if sum_exp_max > 0 else 0.0
        d_max_soft = max_d + (log_sum_exp_max / beta_val)
        
        if d_max_soft < 1e-10:
            return 1e10
            
        ratio = d_min_soft / d_max_soft
        return -ratio * ratio

    try:
        result = minimize(
            lambda x: smooth_metric(x),
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result.success:
            refined = result.x.reshape(14, 3)
            # Ensure finite
            if np.all(np.isfinite(refined)):
                return refined
    except Exception:
        pass
        
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit
    
    pool = _get_candidate_pool()
    n_candidates = len(pool)
    
    # We need to select 14 points from the pool.
    # To maintain antipodal symmetry which is often optimal for such packing problems,
    # we can restrict our search to selecting 7 pairs of antipodal points.
    # However, the pool might not contain perfect antipodes for all points.
    # Let's check for antipodes.
    
    # Build a map of antipodes
    antipode_map = {}
    for i in range(n_candidates):
        p = pool[i]
        ant = -p
        # Find closest candidate to ant
        dists_to_ant = np.linalg.norm(pool - ant, axis=1)
        j = np.argmin(dists_to_ant)
        if dists_to_ant[j] < 1e-5:
            antipode_map[i] = j
        else:
            # If no close antipode, we can't use this in a strict antipodal pair strategy
            # unless we allow asymmetry. 
            # For simplicity and robustness, let's just do a general search 
            # but bias towards symmetric starts.
            pass
            
    # Strategy 1: Select 14 points from pool that maximize min distance.
    # Since 14 is small relative to pool size (~30), we can use a greedy approach
    # followed by local search, or a limited random sampling.
    
    best_points = None
    best_metric = -1.0
    
    # Greedy initialization
    # Start with the point that has the most "room" or just the first one.
    # Better: Start with a random subset and iteratively improve.
    
    # Let's use a few random restarts of a "k-means-like" or "farthest point" sampling
    # constrained to the pool.
    
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Select 14 points randomly? No, that's likely poor.
        # Use Farthest Point Sampling on the pool to get a diverse set of 14.
        
        # Pick a random start point
        current_idx = [rng.randint(n_candidates)]
        current_pts = [pool[current_idx[0]]]
        
        # Compute min distance to existing set for all other points
        min_dists = np.linalg.norm(pool - current_pts[0], axis=1)
        
        for _ in range(13):
            # Mask out already selected
            for idx in current_idx:
                min_dists[idx] = 0.0
                
            # Pick point with max min_dist
            next_idx = np.argmax(min_dists)
            current_idx.append(next_idx)
            current_pts.append(pool[next_idx])
            
            # Update min_dists
            d_to_new = np.linalg.norm(pool - pool[next_idx], axis=1)
            min_dists = np.minimum(min_dists, d_to_new)
            
        candidate = np.array(current_pts)
        
        # Refine this candidate
        refined = _refine_points(candidate, start_time, deadline)
        
        metric = _evaluate(refined)
        
        if metric > best_metric:
            best_metric = metric
            best_points = refined.copy()
            
        # Also try swapping one point to see if we can improve
        # Local search: for each point in the set, try replacing it with a pool point
        # that increases the min distance.
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_set = best_points if (restart == 0 or best_points is None) else refined
            # Actually, we should refine the current best found in this restart
            # Let's refine the current 'refined' point set further
            
            for i in range(14):
                if time.time() > deadline - 5:
                    break
                    
                # Try replacing point i with every point in pool
                best_new_point = None
                best_local_metric = _evaluate(refined)
                
                for j in range(n_candidates):
                    if j in current_idx:
                        continue
                        
                    test_points = refined.copy()
                    test_points[i] = pool[j]
                    
                    # Quick check: does this increase the min distance?
                    # Full evaluation is expensive, but n_candidates is small (~30)
                    # and 14 points, so 14*30 = 420 evaluations per iteration.
                    # pdist is fast.
                    
                    m = _evaluate(test_points)
                    if m > best_local_metric:
                        best_local_metric = m
                        best_new_point = pool[j]
                        
                if best_new_point is not None:
                    refined[i] = best_new_point
                    improved = True
                    # Re-refine after swap
                    refined = _refine_points(refined, start_time, deadline)
                    best_local_metric = _evaluate(refined)
                    
                    if best_local_metric > best_metric:
                        best_metric = best_local_metric
                        best_points = refined.copy()

    # Fallback: Icosahedron + 2 poles
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        
    # Final check
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
