import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_sum_r = 0.0
    best_result = None
    
    # We search over aspect ratios.
    # w + h = 2. Let w = 2t, h = 2(1-t).
    # t is in (0, 1).
    # Symmetry: t and 1-t are equivalent if we swap x and y.
    # We can just search t in [0.3, 0.7] or so.
    aspect_ratios = np.linspace(0.35, 0.65, 15)
    
    # Number of restarts per aspect ratio. 
    # We will adjust this dynamically if time allows, but let's start with a fixed reasonable number.
    # 15 ratios * 20 restarts = 300 optimizations.
    # Each optimization might take ~0.1s to 0.5s. Total ~30-150s. 
    # We need to be careful. Let's use a time-based loop.
    
    def get_rect_dims(t):
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def objective(z, w, h):
        """
        z: flattened array [x1, y1, r1, x2, y2, r2, ...]
        Returns: -sum(r) + penalty
        """
        # Unpack
        coords = z.reshape((N, 3))
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        sum_r = np.sum(rs)
        
        penalty = 0.0
        
        # Boundary penalties
        # Violation if x < r => x - r < 0
        # Violation if x > w - r => w - x - r < 0
        # We use squared penalty for smoothness
        v_left = np.maximum(0, rs - xs)
        v_right = np.maximum(0, rs - (w - xs))
        v_bottom = np.maximum(0, rs - ys)
        v_top = np.maximum(0, rs - (h - ys))
        
        penalty += np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Overlap penalties
        # dist^2 = (xi-xj)^2 + (yi-yj)^2
        # overlap = max(0, ri + rj - dist)
        # penalty += sum(overlap^2)
        
        # Vectorized overlap calculation is tricky, let's loop for clarity or use broadcasting
        # Broadcasting: 
        # dx = xs[:, None] - xs[None, :]
        # dy = ys[:, None] - ys[None, :]
        # dist_sq = dx**2 + dy**2
        # min_dist = rs[:, None] + rs[None, :]
        # overlap = np.maximum(0, min_dist - np.sqrt(dist_sq))
        # penalty += np.sum(overlap**2) / 2 # divide by 2 to avoid double counting? No, sum over all pairs i!=j is 2 * sum over i<j.
        # Since we sum over all i,j including i=j (which is 0), it's fine.
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx**2 + dy**2
        # Avoid sqrt of 0 issues, though sqrt(0)=0 is fine
        dist = np.sqrt(dist_sq)
        min_dist = rs[:, None] + rs[None, :]
        
        # overlap is zero on diagonal
        overlap = np.maximum(0, min_dist - dist)
        
        # Sum of squared overlaps
        # Note: This sums over all i,j. 
        penalty += np.sum(overlap**2)
        
        # Scale penalty to dominate the objective if violations exist
        # If sum_r is around 1.0, penalty should be >> 1.0 if invalid.
        # Let's use a coefficient of 10.
        return -sum_r + 10.0 * penalty

    def check_valid(z, w, h):
        coords = z.reshape((N, 3))
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        # Check boundaries
        if np.any(xs - rs < -1e-5) or np.any(w - xs - rs < -1e-5):
            return False
        if np.any(ys - rs < -1e-5) or np.any(h - ys - rs < -1e-5):
            return False
            
        # Check overlaps
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)
        min_dist = rs[:, None] + rs[None, :]
        
        # Check if any pair (i,j) with i!=j has dist < min_dist - eps
        # np.fill_diagonal(dist, np.inf)
        # if np.any(dist < min_dist - 1e-5):
        #     return False
        
        # More efficient:
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < min_dist[i,j] - 1e-5:
                    return False
        return True

    def initialize_grid(w, h):
        """
        Initialize circles in a grid pattern.
        """
        side = int(np.ceil(np.sqrt(N)))
        # We have N circles, side x side grid
        # w / side, h / side cell size
        cw = w / side
        ch = h / side
        
        z = np.zeros(N * 3)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                # Max radius in cell is min(cw, ch) / 2
                # Add some jitter
                r0 = 0.4 * min(cw, ch)
                
                z[3*idx] = cx + random.uniform(-0.01, 0.01)
                z[3*idx+1] = cy + random.uniform(-0.01, 0.01)
                z[3*idx+2] = r0 + random.uniform(-0.005, 0.005)
                idx += 1
        return z

    # Main search loop
    for t in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w, h = get_rect_dims(t)
        
        # Determine number of restarts based on remaining time
        # Assume ~0.2s per optimization on average
        remaining_time = deadline - time.time()
        max_restarts = int(remaining_time / 0.5) # conservative estimate
        max_restarts = max(5, min(50, max_restarts)) # between 5 and 50 restarts
        
        for k in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            z0 = initialize_grid(w, h)
            
            # Bounds for L-BFGS-B
            # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
            # We use loose bounds and rely on penalty for tight constraints
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0, 0.5 * min(w, h)))
            
            try:
                res = minimize(
                    objective, 
                    z0, 
                    args=(w, h), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res.success:
                    z_opt = res.x
                    # Check validity
                    if check_valid(z_opt, w, h):
                        sum_r = np.sum(z_opt.reshape(N, 3)[:, 2])
                        if sum_r > best_sum_r:
                            best_sum_r = sum_r
                            # Store the result in the required format (N, 3)
                            best_result = z_opt.reshape(N, 3).copy()
                            # print(f"Found better: {best_sum_r:.4f} at t={t:.2f}, restart={k}")
                            
            except Exception:
                pass

    # If no valid result was found (unlikely), return a fallback
    if best_result is None:
        # Fallback: simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                result[idx] = [cx, cy, r]
                idx += 1
        return result

    return best_result
