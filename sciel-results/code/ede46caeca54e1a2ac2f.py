"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for equal-area triangles from 30 points."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s

    def evaluate(points):
        """
        Compute the maximum multiplicity of any positive doubled area.
        Uses vectorized NumPy operations for speed.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        # We need to count areas for all triples (i, j, k) with i < j < k
        # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
        # Doubled Area = |det(P_j - P_i, P_k - P_i)|
        
        # To save memory and time, we can iterate over i, and vectorize j, k.
        # However, n=30 is small enough that a fully vectorized approach might be complex 
        # to implement correctly for unique triples without O(n^3) memory.
        # Instead, we iterate i, and for each i, compute areas for all j > i, k > j.
        
        # A faster approach:
        # For each pair (i, j), the vector v = P_j - P_i.
        # For each k > j, area = |cross(v, P_k - P_i)|.
        
        # Let's try a loop over i, and vectorized inner steps.
        areas = []
        for i in range(n - 2):
            xi, yi = pts[i]
            # Vectors from i to all other points
            # We only care about j > i and k > j
            # This is tricky to fully vectorize for unique triples without masking.
            
            # Alternative: Generate all pairs (j, k) with j > i, k > j
            # For fixed i:
            # Indices j in [i+1, n-1], k in [j+1, n-1]
            
            # Let's just use a tight loop with NumPy for the inner part if possible, 
            # or stick to the previous method but optimized.
            
            # Actually, the previous method in the prompt used a loop over i, j and 
            # vectorized k. Let's use that.
            for j in range(i + 1, n - 1):
                xj, yj = pts[j]
                dx = xj - xi
                dy = yj - yi
                
                # Vector for all k > j
                # P_k - P_i
                # We can precompute P_k - P_i for all k? No, it depends on i.
                
                # Compute cross product for all k > j
                # P_k = pts[k]
                # v1 = P_j - P_i = (dx, dy)
                # v2 = P_k - P_i = (pts[k][0] - xi, pts[k][1] - yi)
                # cross = dx * (pts[k][1] - yi) - dy * (pts[k][0] - xi)
                
                # Slice points from j+1 to end
                sub_pts = pts[j+1:]
                vx = sub_pts[:, 0] - xi
                vy = sub_pts[:, 1] - yi
                
                cross = dx * vy - dy * vx
                abs_cross = np.abs(cross)
                
                # Filter out zero areas (collinear)
                # We only count positive areas
                valid = abs_cross[abs_cross > 0]
                if len(valid) > 0:
                    areas.append(valid)
        
        if not areas:
            return 0
            
        # Concatenate all areas
        all_areas = np.concatenate(areas)
        
        # Count frequencies
        # np.unique is efficient
        _, counts = np.unique(all_areas, return_counts=True)
        return int(counts.max())

    def try_config(points, current_best_score):
        """Try a configuration and return updated best if improved."""
        # Check distinctness
        if len(set(points)) != 30:
            return None, current_best_score
        # Check bounds
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in points):
            return None, current_best_score
        score = evaluate(points)
        if score > current_best_score:
            return points, score
        return None, current_best_score

    # Initial configuration: a simple grid
    # 6x5 grid
    best_points = [(i, j) for i in range(6) for j in range(5)]
    best_score = evaluate(best_points)

    # Phase 1: Systematic Grid Search
    # Try various grid shapes and spacings
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    for w, h in factors_30:
        # Spacings
        max_sx = 30 // (w - 1) if w > 1 else 60
        max_sy = 30 // (h - 1) if h > 1 else 60
        
        # Limit search space to reasonable values
        for sx in range(1, min(max_sx, 10) + 1):
            for sy in range(1, min(max_sy, 10) + 1):
                if time.time() > deadline:
                    return best_points
                
                # Generate grid points
                # Center them to stay in [-30, 30]
                width = (w - 1) * sx
                height = (h - 1) * sy
                
                # If width/height too big, skip or scale down? 
                # The problem asks for integer coordinates in [-30, 30].
                # If the grid is too wide, it won't fit.
                if width > 60 or height > 60:
                    continue
                
                cx = -width // 2
                cy = -height // 2
                
                pts = [(cx + i * sx, cy + j * sy) for i in range(w) for j in range(h)]
                
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Phase 2: Sheared Grids
    # Points (i*sx + j*sxy, j*sy)
    for w, h in [(6,5), (5,6), (10,3), (3,10)]:
        for sx in range(1, 6):
            for sy in range(1, 6):
                for sxy in range(-3, 4):
                    if time.time() > deadline:
                        return best_points
                    
                    pts = [(i * sx + j * sxy, j * sy) for i in range(w) for j in range(h)]
                    
                    # Check bounds
                    max_x = max(p[0] for p in pts)
                    min_x = min(p[0] for p in pts)
                    max_y = max(p[1] for p in pts)
                    min_y = min(p[1] for p in pts)
                    
                    if max_x > 30 or min_x < -30 or max_y > 30 or min_y < -30:
                        continue
                        
                    result, new_score = try_config(pts, best_score)
                    if result is not None:
                        best_points = result
                        best_score = new_score

    # Phase 3: Local Search (Hill Climbing)
    # Since evaluate is fast, we can do many iterations.
    random.seed(42)
    
    improved = True
    max_iterations = 1000 # High limit, controlled by time
    iteration = 0
    
    while improved and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        # Shuffle points to avoid bias in update order? 
        # Or just iterate through indices.
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
            
            old_x, old_y = best_points[idx]
            
            # Try moves in a neighborhood
            # Radius 3-5 is usually good
            radius = 4 
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    
                    # Check distinctness
                    if len(set(new_points)) != 30:
                        continue
                    
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Phase 4: Random Restarts with Perturbation
    if time.time() < deadline - 5:
        random.seed(123)
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            
            # Create a perturbed version of the best solution
            new_points = list(best_points)
            num_moves = random.randint(1, 5)
            for _ in range(num_moves):
                idx = random.randint(0, 29)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_x = new_points[idx][0] + dx
                new_y = new_points[idx][1] + dy
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_points[idx] = (new_x, new_y)
            
            # Check validity
            if len(set(new_points)) != 30:
                continue
            if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in new_points):
                continue
                
            result, new_score = try_config(new_points, best_score)
            if result is not None:
                best_points = result
                best_score = new_score

    # Phase 5: Final Polish
    # Run one more pass of local search with smaller steps (radius 1-2) to settle
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(30):
                if time.time() > deadline - 1:
                    break
                old_x, old_y = best_points[idx]
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(best_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        score = evaluate(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_points
# EVOLVE-BLOCK-END
