import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_score(points):
    """
    Compute the packing radius for a set of points.
    r = min( min_{i<j} d_ij / 2, min_i (1 - |c_i|) )
    """
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    if np.any(np.isnan(points)):
        return -np.inf
        
    # Boundary constraints
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    # Pairwise constraints
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_obj(x_flat, beta):
    """
    Objective: -soft_min(distances)
    Distances include pairwise (d_ij) and boundary (1-|c_i|).
    Note: For pairwise, the constraint is d_ij >= 2r.
    To handle this uniformly with soft-min, we can scale pairwise distances by 1/2 
    or just define the "effective radius" directly.
    
    Let's define the "effective clearance" for each constraint:
    - Boundary: 1 - |c_i|
    - Pairwise: d_ij / 2
    
    We want to maximize min(clearances).
    """
    points = x_flat.reshape(-1, 2)
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[i_idx, j_idx]
    # Avoid sqrt for gradient? No, we need gradient.
    # dist = sqrt(sq). d(dist)/dx = (x_i - x_j) / dist
    pair_dists = np.sqrt(pair_dists_sq)
    pair_dists_safe = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    radii_safe = np.where(radii > 1e-12, radii, 1.0)
    bound_clearances = 1.0 - radii
    
    # Clearances array
    # We want to maximize min( pair_dists/2, bound_clearances )
    # Let's combine them into a single vector of "clearances"
    pair_clearances = pair_dists / 2.0
    clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Soft min
    # soft_min = -1/beta * log(sum(exp(-beta * x)))
    y = -beta * clearances
    y_max = np.max(y)
    exp_y = np.exp(y - y_max)
    sum_exp = np.sum(exp_y)
    
    if sum_exp == 0:
        val = -np.inf
        grad_clear = np.zeros_like(clearances)
    else:
        val = -(np.log(sum_exp) + y_max) / beta
        grad_clear = exp_y / sum_exp
        
    obj = -val
    
    # Gradient
    grad_points = np.zeros_like(points)
    
    n_pairs = len(pair_clearances)
    grad_pair = grad_clear[:n_pairs]
    grad_bound = grad_clear[n_pairs:]
    
    # Boundary gradient
    # d(1 - |c_i|)/dx_i = - c_i / |c_i|
    grad_bound_vec = -points / radii_safe[:, np.newaxis]
    grad_points += grad_bound[:, np.newaxis] * grad_bound_vec
    
    # Pairwise gradient
    # d(||c_i - c_j||/2)/dx_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    diff_vec = points[i_idx] - points[j_idx]
    grad_diff = 0.5 * diff_vec / pair_dists_safe[:, np.newaxis]
    
    np.add.at(grad_points, i_idx, grad_pair[:, np.newaxis] * grad_diff)
    np.add.at(grad_points, j_idx, -grad_pair[:, np.newaxis] * grad_diff)
    
    grad_flat = -grad_points.flatten()
    
    return obj, grad_flat

def _generate_hex_initial(r_target=0.125):
    """
    Generate a hexagonal lattice packing.
    """
    points = []
    # Hexagonal lattice spacing s = 2 * r_target
    s = 2 * r_target
    h = s * np.sqrt(3) / 2
    
    # Generate points in a square region large enough to cover unit circle
    # Max coordinate is 1.0
    # Number of points needed is roughly (2/s)^2 * (2/pi) * pi/2... 
    # Just generate a grid and filter.
    
    # Rows
    n_rows = int(2.0 / h) + 2
    for i in range(-n_rows, n_rows + 1):
        y = i * h
        if abs(y) > 1.0 + s:
            continue
        # Offset x for even/odd rows
        offset = 0.0 if i % 2 == 0 else s / 2.0
        n_cols = int(2.0 / s) + 2
        for j in range(-n_cols, n_cols + 1):
            x = j * s + offset
            if abs(x) > 1.0 + s:
                continue
            # Check if inside circle with margin
            if x*x + y*y <= (1.0 - r_target)**2:
                points.append([x, y])
                
    points = np.array(points)
    
    # We need exactly 50 points.
    # If we have more, we need to select 50 that are well distributed.
    # If we have fewer, we need to add more (maybe loosen r_target).
    
    if len(points) < N:
        # Try a smaller radius to get more points
        return _generate_hex_initial(r_target * 0.9)
        
    if len(points) > N:
        # Select N points.
        # Strategy: Pick the ones closest to a uniform distribution.
        # Simple heuristic: Sort by distance from origin? No, that clusters center.
        # Better: Random selection with rejection? Or just take the first N?
        # Taking first N from the loop order (row by row) might be biased.
        # Let's pick points that maximize the minimum distance to already picked points (greedy).
        
        selected = [points[0]]
        remaining = list(points[1:])
        
        # Greedy selection
        while len(selected) < N and remaining:
            # Find point in remaining that maximizes min distance to selected
            best_idx = -1
            best_min_dist = -1
            for idx, p in enumerate(remaining):
                dists = np.linalg.norm(np.array(selected) - p, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
                    
            if best_idx == -1:
                break
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
            
        if len(selected) < N:
             # Fill with random points from remaining
             np.random.shuffle(remaining)
             selected.extend(remaining[:N - len(selected)])
             
        points = np.array(selected)
        
    return points

def _polish(points, time_limit=5.0):
    """
    Use Nelder-Mead to polish the result.
    """
    start = time.time()
    
    def obj(x):
        p = x.reshape(-1, 2)
        return -_compute_score(p)
        
    try:
        res = minimize(
            obj, 
            points.flatten(), 
            method='Nelder-Mead', 
            options={
                'maxiter': 2000, 
                'xatol': 1e-10, 
                'fatol': 1e-10, 
                'adaptive': True
            }
        )
        if time.time() - start < time_limit:
            new_points = res.x.reshape(-1, 2)
            if _compute_score(new_points) >= _compute_score(points) - 1e-12:
                return new_points
    except:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # 1. Generate initial hexagonal structure
    init_points = _generate_hex_initial(r_target=0.125)
    
    # Ensure we have exactly 50 points
    if len(init_points) != N:
        # Fallback: random points
        init_points = np.random.rand(N, 2) * 2 - 1
        init_points = init_points * (1.0 / np.linalg.norm(init_points, axis=1)[:, np.newaxis]) * 0.9
        # This might not be great, but it's a valid start.
        # Actually, let's just pad/truncate if necessary
        if len(init_points) < N:
            # Add random points
            extra = np.random.rand(N - len(init_points), 2) * 0.5
            init_points = np.vstack([init_points, extra])
        else:
            init_points = init_points[:N]
            
    best_points = init_points.copy()
    best_score = _compute_score(best_points)
    
    # 2. Annealing optimization with L-BFGS-B
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    
    current_points = init_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        x0 = current_points.flatten()
        
        try:
            res = minimize(
                _soft_min_obj, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 500, 
                    'ftol': 1e-12, 
                    'gtol': 1e-8,
                    'maxls': 30
                }
            )
            
            new_points = res.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                current_points = new_points.copy()
            else:
                # Even if it didn't improve the hard score, it might be better for the soft objective
                # and lead to a better solution later.
                # However, if it got worse, we should probably stick to the best known.
                # Let's check if the soft objective improved.
                # Actually, if the hard score didn't improve, it's risky to move away from best_points.
                # But sometimes a slight decrease in hard score allows a larger jump later.
                # For now, let's only update current_points if it's not significantly worse.
                if new_score > best_score - 1e-4:
                    current_points = new_points.copy()
                    
        except Exception as e:
            pass
            
        if time.time() > deadline - 20.0:
            break
            
    # 3. Polish step
    if time.time() < deadline - 5.0:
        polished = _polish(best_points, time_limit=10.0)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    return best_points
