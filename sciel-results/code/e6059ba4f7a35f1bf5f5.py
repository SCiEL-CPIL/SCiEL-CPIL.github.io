import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial seed: 6x5 grid
    def get_initial_grid(w, h):
        return [(x, y) for x in range(w) for y in range(h)]
    
    best_points = get_initial_grid(6, 5)
    
    def score_numpy(points):
        """Compute the maximum multiplicity of any single positive doubled area using NumPy."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        # Compute all triple determinants
        # For points i, j, k: det = |xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)|
        # This equals 2 * area of triangle
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    xk, yk = pts[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        cnt = Counter(areas)
        return max(cnt.values())
    
    def score_numpy_fast(points):
        """Faster score using vectorized NumPy operations."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        # Precompute pairwise differences
        # For each triple (i,j,k), det = xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj
        # = xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)
        
        # Use broadcasting to compute all triple determinants
        # This creates a 3D array which might be memory heavy for n=30
        # n=30: C(30,3) = 4060 triples, manageable
        
        # Alternative: compute all pairwise cross products
        # For vectors v1 = p_j - p_i, v2 = p_k - p_i, det = v1_x * v2_y - v1_y * v2_x
        # But we need absolute value and to count multiplicities
        
        areas = []
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                xj, yj = pts[j]
                dx_ij = xj - xi
                dy_ij = yj - yi
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    dx_ik = xk - xi
                    dy_ik = yk - yi
                    d = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        cnt = Counter(areas)
        return max(cnt.values())
    
    def score_with_counter(points, target_area=None):
        """Score focusing on target area if provided, else max multiplicity."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        cnt = Counter()
        target_count = 0
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                xj, yj = pts[j]
                dx_ij = xj - xi
                dy_ij = yj - yi
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    dx_ik = xk - xi
                    dy_ik = yk - yi
                    d = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                    if d > 0:
                        cnt[d] += 1
                        if target_area is not None and d == target_area:
                            target_count += 1
        if target_area is not None:
            return target_count
        return max(cnt.values()) if cnt else 0
    
    # Evaluate seed
    best_score = score_numpy_fast(best_points)
    
    # Strategy: Try different target areas and use simulated annealing
    # Common doubled areas for grid points: 2, 4, 6, 8, 10, etc.
    # For a 6x5 grid, let's compute what areas are common
    
    # First, let's try a smarter initial configuration
    # Points that lie on a line or near a line will have small areas
    # We want many triangles with the SAME area
    
    # Try configurations based on arithmetic progressions
    # If points are (a*i, b*j) for i in range(m), j in range(n)
    # Then triangle areas depend on the differences
    
    # Let's try multiple initial configurations and pick the best
    candidates = []
    
    # Grid 6x5
    candidates.append(get_initial_grid(6, 5))
    
    # Grid 5x6
    candidates.append(get_initial_grid(5, 6))
    
    # Try a configuration with more structure
    # Points on a parabola: (i, i^2) scaled
    # But need to stay in [-30, 30]
    for scale in [1, 2, 3]:
        pts = []
        for i in range(-10, 10):
            x = i * scale
            y = (i * i) // 4  # Scale down to fit
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) >= 30:
            candidates.append(pts[:30])
    
    # Try rectangular grids of various sizes
    for w in range(4, 8):
        for h in range(4, 8):
            if w * h >= 30:
                pts = []
                for x in range(-w//2, w//2 + 1):
                    for y in range(-h//2, h//2 + 1):
                        pts.append((x, y))
                        if len(pts) == 30:
                            break
                    if len(pts) == 30:
                        break
                if len(pts) == 30:
                    candidates.append(pts)
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) == 30 and len(set(cand)) == 30:
            s = score_numpy_fast(cand)
            if s > best_score:
                best_score = s
                best_points = cand
    
    # Now use simulated annealing with target area focus
    # Pick a target area from the current best configuration
    pts_arr = np.array(best_points, dtype=np.int64)
    n = 30
    area_counts = Counter()
    for i in range(n):
        xi, yi = pts_arr[i]
        for j in range(i + 1, n):
            xj, yj = pts_arr[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts_arr[k]
                dx_ik = xk - xi
                dy_ik = yk - yi
                d = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                if d > 0:
                    area_counts[d] += 1
    
    # Pick the most common area as target
    if area_counts:
        target_area = area_counts.most_common(1)[0][0]
    else:
        target_area = 2
    
    # Simulated annealing
    current_points = best_points[:]
    current_score = best_score
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    
    # Precompute for faster evaluation
    def quick_score(points, target=None):
        """Quick score for annealing."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        if target is not None:
            count = 0
            for i in range(n):
                xi, yi = pts[i]
                for j in range(i + 1, n):
                    xj, yj = pts[j]
                    dx_ij = xj - xi
                    dy_ij = yj - yi
                    for k in range(j + 1, n):
                        xk, yk = pts[k]
                        dx_ik = xk - xi
                        dy_ik = yk - yi
                        d = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                        if d == target:
                            count += 1
            return count
        else:
            cnt = Counter()
            for i in range(n):
                xi, yi = pts[i]
                for j in range(i + 1, n):
                    xj, yj = pts[j]
                    dx_ij = xj - xi
                    dy_ij = yj - yi
                    for k in range(j + 1, n):
                        xk, yk = pts[k]
                        dx_ik = xk - xi
                        dy_ik = yk - yi
                        d = abs(dx_ij * dy_ik - dy_ij * dx_ik)
                        if d > 0:
                            cnt[d] += 1
            return max(cnt.values()) if cnt else 0
    
    # Annealing loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a point to move
        idx = random.randint(0, 29)
        
        # Try several random positions and pick the best
        best_move_score = -1
        best_new_point = None
        
        for _ in range(3):  # Try 3 random positions per iteration
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            
            # Check distinctness
            is_dup = False
            for pi, pj in enumerate(current_points):
                if pi != idx and pj == new_point:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            candidate = current_points[:]
            candidate[idx] = new_point
            s = quick_score(candidate, target_area)
            
            if s > best_move_score:
                best_move_score = s
                best_new_point = new_point
        
        if best_new_point is None:
            continue
        
        candidate = current_points[:]
        candidate[idx] = best_new_point
        s = best_move_score
        
        # Accept if better, or with probability exp(-delta/T)
        if s > current_score:
            current_points = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_points = current_points[:]
        else:
            delta = current_score - s
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / max(T, 0.01))
            if accept:
                current_points = candidate
                current_score = s
        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
    
    # Final polish: try small perturbations around best solution
    for _ in range(5000):
        if time.time() > deadline - 1:
            break
        
        idx = random.randint(0, 29)
        old_point = best_points[idx]
        
        # Try nearby positions
        tried = set()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_point[0] + dx
                new_y = old_point[1] + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                new_point = (new_x, new_y)
                if new_point in tried:
                    continue
                tried.add(new_point)
                
                # Check distinctness
                is_dup = False
                for pi, pj in enumerate(best_points):
                    if pi != idx and pj == new_point:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                candidate = best_points[:]
                candidate[idx] = new_point
                s = quick_score(candidate, target_area)
                
                if s > best_score:
                    best_score = s
                    best_points = candidate
    
    return best_points
