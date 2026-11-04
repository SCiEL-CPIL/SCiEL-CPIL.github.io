import numpy as np
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
    
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=100, tol=1e-8):
    """
    Vectorized relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        # Vectorized clamp
        pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
        pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
        # 2. Pairwise collisions: push apart (Vectorized)
        # Compute pairwise differences
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist_sq = dx**2 + dy**2
        # Avoid division by zero
        dist = np.sqrt(dist_sq + 1e-12)
        
        min_dist = rad[:, None] + rad[None, :]
        
        # Overlap amount: positive if overlapping
        overlap = np.maximum(0, min_dist - dist)
        
        # If any overlap > tol, we moved
        if np.max(overlap) > tol:
            moved = True
            
            # Unit vectors for separation
            # Where dist is very small, direction is undefined, but overlap is handled by random perturbation or just zero force?
            # To keep it simple and fast, we use the direction defined by dx/dy.
            # For dist < eps, we can set force to 0 and rely on boundary or random jitter if needed, 
            # but for packing, distinct centers are usually maintained by initial grid.
            # Let's use safe division.
            inv_dist = np.zeros_like(dist)
            mask = dist > 1e-8
            inv_dist[mask] = 1.0 / dist[mask]
            
            nx = dx * inv_dist
            ny = dy * inv_dist
            
            # Force on i from j is -nx * overlap * 0.5
            # We sum forces from all j on i
            # Force vector for i: sum_j (-nx[i,j] * overlap[i,j] * 0.5)
            # Note: overlap[i,j] = overlap[j,i]
            # nx[i,j] = -nx[j,i]
            
            force_x = -0.5 * np.sum(nx * overlap, axis=1)
            force_y = -0.5 * np.sum(ny * overlap, axis=1)
            
            pos[:, 0] += force_x
            pos[:, 1] += force_y
            
            # Re-clamp to boundary after movement
            pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
            pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
            
    # Final clamp to ensure boundary validity
    pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
    pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
    return np.column_stack((pos, rad))

def _jittered_grid_init(w, h, n_circles, seed=42):
    """
    Generate initial circle positions using a jittered grid.
    This is often more robust than hexagonal for varying aspect ratios.
    """
    np.random.seed(seed)
    circles = np.zeros((n_circles, 3))
    
    # Estimate grid dimensions
    # We want roughly n_circles cells.
    # Adjust aspect ratio to find best grid fit
    ar = w / h
    if ar >= 1:
        cols = int(np.sqrt(n_circles * ar))
        rows = int(np.ceil(n_circles / cols))
    else:
        rows = int(np.sqrt(n_circles / ar))
        cols = int(np.ceil(n_circles / rows))
        
    # Calculate cell size
    cell_w = w / cols
    cell_h = h / rows
    
    # Base radius is half the min cell dimension, slightly reduced for safety
    r_base = 0.45 * min(cell_w, cell_h)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_circles:
                break
            # Center of cell
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            
            # Add jitter
            jitter_x = np.random.uniform(-0.2, 0.2) * cell_w
            jitter_y = np.random.uniform(-0.2, 0.2) * cell_h
            
            x = cx + jitter_x
            y = cy + jitter_y
            
            # Clamp position to keep circle inside bounds (roughly)
            x = np.clip(x, r_base, w - r_base)
            y = np.clip(y, r_base, h - r_base)
            
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = r_base
            idx += 1
            
    # If we didn't fill all, duplicate last or just use what we have?
    # The problem requires exactly N circles.
    # If grid is smaller than N, we need to split cells or add more.
    # But usually cols*rows >= N.
    # If cols*rows < N, we increase resolution.
    if idx < n_circles:
        # Fallback: simple random scatter
        for i in range(idx, n_circles):
            circles[i, 0] = np.random.uniform(0, w)
            circles[i, 1] = np.random.uniform(0, h)
            circles[i, 2] = r_base * 0.5
            
    return circles

def _local_search_optimize(w, h, deadline, seed=42):
    """
    Local Search to maximize sum of radii while maintaining validity.
    Optimized for speed with vectorized collision resolution.
    """
    np.random.seed(seed)
    
    # Initial state
    circles = _jittered_grid_init(w, h, N, seed=seed)
    circles = _resolve_collisions(circles, w, h, max_iters=50)
    
    if not _validate_solution(circles, w, h, N):
        return None
        
    current_circles = circles
    current_score = np.sum(circles[:, 2])
    
    best_circles = current_circles.copy()
    best_score = current_score
    
    n = N
    no_improve_count = 0
    max_no_improve = 100 # Reduced to fail faster and try other strategies
    
    # Time check inside loop
    while time.time() < deadline - 2.0:
        if no_improve_count > max_no_improve:
            break
            
        # Strategy 1: Expand a random circle
        idx = np.random.choice(n)
        new_r = current_circles[idx, 2] * 1.05 # 5% increase for faster progress
        new_r = min(new_r, min(w, h) / 2.0)
        
        new_circles = current_circles.copy()
        new_circles[idx, 2] = new_r
        
        # Resolve collisions with fewer iterations
        new_circles = _resolve_collisions(new_circles, w, h, max_iters=20)
        
        if _validate_solution(new_circles, w, h, N):
            new_score = np.sum(new_circles[:, 2])
            if new_score > current_score:
                current_circles = new_circles
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_circles = current_circles.copy()
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Strategy 2: Move a random circle (reposition)
        if np.random.rand() < 0.5:
            idx = np.random.choice(n)
            r = current_circles[idx, 2]
            new_x = np.random.uniform(r, w - r)
            new_y = np.random.uniform(r, h - r)
            
            new_circles = current_circles.copy()
            new_circles[idx, 0] = new_x
            new_circles[idx, 1] = new_y
            
            new_circles = _resolve_collisions(new_circles, w, h, max_iters=20)
            
            if _validate_solution(new_circles, w, h, N):
                new_score = np.sum(new_circles[:, 2])
                if new_score > current_score:
                    current_circles = new_circles
                    current_score = new_score
                    no_improve_count = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = current_circles.copy()
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
    return best_circles

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin under 120s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Reduced list to ensure we finish within time
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 1.1, 0.9, 1.3, 0.75, 1.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Run Local Search for this aspect ratio
        result = _local_search_optimize(w, h, deadline, seed=int(ar*100))
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _jittered_grid_init(w, h, N)
        best_result = _resolve_collisions(best_result, w, h, max_iters=50)
        if not _validate_solution(best_result, w, h, N):
            # Last resort: tiny circles
            best_result = np.zeros((N, 3))
            for i in range(N):
                x = (i % 5 + 0.5) * 0.2
                y = (i // 5 + 0.5) * 0.2
                best_result[i] = [x, y, 0.05]
            best_result = _resolve_collisions(best_result, w, h, max_iters=50)
            if not _validate_solution(best_result, w, h, N):
                 # Absolute last resort: return a valid but small solution
                 best_result = np.zeros((N, 3))
                 best_result[:, 0] = np.linspace(0.1, 0.9, N)
                 best_result[:, 1] = 0.5
                 best_result[:, 2] = 0.01
                 best_result = _resolve_collisions(best_result, w, h, max_iters=10)

    return best_result
