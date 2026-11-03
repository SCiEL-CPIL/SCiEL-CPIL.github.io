"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    N = 21
    # Perimeter 4 => w + h = 2.
    # We want to maximize sum(r_i).
    # For a rectangular packing of N circles, a long thin rectangle is often better 
    # for maximizing sum of radii because the "waste" space is minimized relative to perimeter.
    # However, we must fit N circles.
    
    best_result = None
    best_value = -np.inf

    def validate(circles, w, h):
        """Check if circles are within bounds and non-overlapping."""
        if circles is None:
            return False
        # Bounds check with small tolerance
        if np.any(circles[:, 0] < -1e-9) or np.any(circles[:, 0] > w + 1e-9):
            return False
        if np.any(circles[:, 1] < -1e-9) or np.any(circles[:, 1] > h + 1e-9):
            return False
        if np.any(circles[:, 2] <= 0):
            return False
        
        # Overlap check
        # Using vectorized distance calculation for speed
        # We need pairwise distances
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Create pairwise difference matrices
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Minimum required distance
        min_dists = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        # Check if any pair (i,j) with i!=j has dist < min_dist - eps
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # If any off-diagonal distance is less than sum of radii (minus tiny epsilon), it's invalid
        if np.any(dists[mask] < min_dists[mask] - 1e-8):
            return False
        return True

    def relax_packing(circles, w, h, iterations=1000, radius_growth_rate=0.001):
        """
        Iteratively adjust positions and grow radii.
        1. Resolve overlaps by pushing circles apart.
        2. Grow radii slightly.
        3. Repeat.
        """
        circles = circles.copy()
        n = len(circles)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
                
            # 1. Resolve overlaps
            # Simple iterative repulsion
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    
                    if dist < min_dist and dist > 1e-9:
                        # Push apart
                        overlap = min_dist - dist
                        # Distribute push based on inverse mass (here just equal)
                        push = overlap / 2.0
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        
                        circles[i, 0] += push * dx_norm
                        circles[i, 1] += push * dy_norm
                        circles[j, 0] -= push * dx_norm
                        circles[j, 1] -= push * dy_norm
                        moved = True
                    elif dist < min_dist:
                        # If centers are too close, push randomly or along x
                        circles[i, 0] += 1e-6
                        circles[j, 0] -= 1e-6
                        moved = True
            
            # 2. Enforce bounds
            for i in range(n):
                r = circles[i, 2]
                if circles[i, 0] < r:
                    circles[i, 0] = r
                if circles[i, 0] > w - r:
                    circles[i, 0] = w - r
                if circles[i, 1] < r:
                    circles[i, 1] = r
                if circles[i, 1] > h - r:
                    circles[i, 1] = h - r
            
            # 3. Grow radii
            # We can grow all radii by a small amount, but we must ensure we don't break validity.
            # A safer approach: grow radii by a very small epsilon, then resolve overlaps again.
            # To be efficient, we only grow if we had space in the previous iteration.
            # For now, let's just try to grow by a fixed small amount and let the repulsion handle it.
            # If repulsion pushes them out of bounds, we shrink back or stop.
            
            # Heuristic: Grow radii if the packing seems "loose".
            # Calculate min clearance.
            # This is expensive. Instead, just grow slightly and rely on bounds/repulsion.
            # If we exceed bounds, we clamp. If we overlap, we push.
            # This might lead to a local maximum where circles are stuck against walls.
            
            # Let's try a different growth strategy: 
            # Calculate the maximum possible radius increase for each circle given its neighbors.
            # This is complex. 
            # Alternative: Start with small radii, grow uniformly, repel.
            
            # For this specific function, let's just do repulsion and bounds.
            # The caller will try different initial radii.
            
            if not moved:
                # If no overlaps, we can try to grow radii
                # Check if we can grow by epsilon
                can_grow = True
                test_r = circles[:, 2] + radius_growth_rate
                # Check bounds
                if np.any(test_r > circles[:, 0]) or np.any(test_r > w - circles[:, 0]) or \
                   np.any(test_r > circles[:, 1]) or np.any(test_r > h - circles[:, 1]):
                    can_grow = False
                else:
                    # Check overlaps with new radii
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = circles[i, 0] - circles[j, 0]
                            dy = circles[i, 1] - circles[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < test_r[i] + test_r[j] - 1e-9:
                                can_grow = False
                                break
                        if not can_grow:
                            break
                
                if can_grow:
                    circles[:, 2] += radius_growth_rate
                else:
                    # Local maximum reached
                    break

        return circles

    # Try different aspect ratios and grid layouts
    # w + h = 2
    # Aspect ratio ar = w/h => w = 2*ar/(1+ar), h = 2/(1+ar)
    
    # Layout 1: 1x21 (Single row)
    # Layout 2: 2x11 (Two rows, staggered)
    # Layout 3: 3x7 (Three rows, staggered)
    
    aspect_ratios = [10.0, 5.0, 3.0, 2.0, 1.5, 1.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Determine grid layout based on aspect ratio
        # If w is large, 1 row is good. If w is moderate, 2 or 3 rows.
        # Let's define a few specific topologies.
        
        topologies = []
        
        # Topology A: 1x21
        # Only viable if w is large enough.
        # Min width for 1x21 circles of radius r: 21*2r + 2r (walls) = 44r? No.
        # Centers from r to w-r. Distance between centers 2r.
        # Total span = 20 * 2r = 40r.
        # w >= 40r + 2r = 42r => r <= w/42.
        # Sum r = 21 * w/42 = w/2.
        # If w=2, h=0 (degenerate). If w=1.9, h=0.1.
        # r <= 1.9/42 ~ 0.045. Sum ~ 0.95.
        
        # Topology B: 2x11
        # Row 0: 11 circles. Row 1: 11 circles (one missing? No, 21 total).
        # Maybe 11 in row 0, 10 in row 1? Or 11 and 11 with one shifted?
        # Let's stick to regular grids for initialization.
        
        # Let's just try a few fixed grid configurations:
        # 1. 1x21
        # 2. 2x11 (11 in top, 10 in bottom staggered)
        # 3. 3x7 (7 in each row)
        
        # --- Config 1: 1x21 ---
        if w > 1.0: # Reasonable for 1 row
            init_circles = np.zeros((N, 3))
            r_init = w / (2 * N + 2) * 0.9 # Slightly smaller to allow movement
            for i in range(N):
                init_circles[i, 0] = r_init + i * 2 * r_init
                init_circles[i, 1] = h / 2
                init_circles[i, 2] = r_init
            
            relaxed = relax_packing(init_circles, w, h, iterations=200, radius_growth_rate=0.0005)
            if validate(relaxed, w, h):
                val = np.sum(relaxed[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = relaxed.copy()

        # --- Config 2: 2x11 (Staggered) ---
        # 11 in row 0, 10 in row 1.
        # This is better for w/h ~ 2-5.
        if w > 0.8:
            init_circles = np.zeros((N, 3))
            # Approx radius
            # Row 0: 11 circles. Span ~ 10*2r. w >= 10*2r + 2r => r <= w/22
            # Row 1: 10 circles. Staggered.
            r_init = min(w / 22, h / 4) * 0.9 # h/4 because 2 rows, radius can be up to h/2
            
            idx = 0
            # Row 0: 11 circles
            for i in range(11):
                cx = r_init + i * 2 * r_init
                cy = r_init + 0.1 * r_init # Slightly above center
                cx = np.clip(cx, r_init, w - r_init)
                init_circles[idx, 0] = cx
                init_circles[idx, 1] = cy
                init_circles[idx, 2] = r_init
                idx += 1
            
            # Row 1: 10 circles
            # Shifted by r_init
            for i in range(10):
                cx = r_init + r_init + i * 2 * r_init
                cy = h - r_init - 0.1 * r_init
                cx = np.clip(cx, r_init, w - r_init)
                init_circles[idx, 0] = cx
                init_circles[idx, 1] = cy
                init_circles[idx, 2] = r_init
                idx += 1
            
            relaxed = relax_packing(init_circles, w, h, iterations=200, radius_growth_rate=0.0005)
            if validate(relaxed, w, h):
                val = np.sum(relaxed[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = relaxed.copy()

        # --- Config 3: 3x7 (Staggered) ---
        # 7 in each row.
        if w > 0.5:
            init_circles = np.zeros((N, 3))
            r_init = min(w / 14, h / 6) * 0.9
            
            idx = 0
            for row in range(3):
                for i in range(7):
                    # Stagger odd rows
                    offset = 0 if row % 2 == 0 else r_init
                    cx = r_init + offset + i * 2 * r_init
                    cy = r_init + row * 2 * r_init
                    cx = np.clip(cx, r_init, w - r_init)
                    cy = np.clip(cy, r_init, h - r_init)
                    init_circles[idx, 0] = cx
                    init_circles[idx, 1] = cy
                    init_circles[idx, 2] = r_init
                    idx += 1
            
            relaxed = relax_packing(init_circles, w, h, iterations=200, radius_growth_rate=0.0005)
            if validate(relaxed, w, h):
                val = np.sum(relaxed[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = relaxed.copy()

    # Fallback: Simple 3x7 grid with conservative radius
    if best_result is None or best_value < 1.0:
        w, h = 1.4, 0.6
        cells = []
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * w / 7
                cy = (i + 0.5) * h / 3
                # Radius limited by cell size
                r = min(w/7, h/3) / 2 * 0.9
                cells.append([cx, cy, r])
        best_result = np.array(cells)
        best_value = np.sum(best_result[:, 2])

    return best_result
# EVOLVE-BLOCK-END
