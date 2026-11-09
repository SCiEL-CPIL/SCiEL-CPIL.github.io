import time
import random
import numpy as np
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area using NumPy."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all C(n,3) triples efficiently
    # For each triple (i,j,k), doubled area = |cross(pts[j]-pts[i], pts[k]-pts[i])|
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Vector from i to j and i to k
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                d = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if d > 0:
                    areas.append(int(d))
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _score_np(points):
    """Faster vectorized scoring using NumPy broadcasting."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Use the formula: doubled area of (i,j,k) = |x_i(y_j-y_k) + x_j(y_k-y_i) + x_k(y_i-y_j)|
    # We can compute this for all triples using broadcasting
    # But C(30,3) = 4060, so let's just do it directly but with numpy arrays
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Generate all index triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    xi = xs[i_idx]
    yi = ys[i_idx]
    xj = xs[j_idx]
    yj = ys[j_idx]
    xk = xs[k_idx]
    yk = ys[k_idx]
    
    # Doubled area = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    doubled = np.abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
    
    # Filter out zero areas
    positive = doubled[doubled > 0]
    if len(positive) == 0:
        return 0
    
    # Count frequencies
    vals, counts = np.unique(positive, return_counts=True)
    return int(counts.max())

def _valid(pts):
    """Check if points form a valid configuration."""
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    for x, y in pts:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
    return True

def _structured_candidates():
    """Generate a variety of structured point configurations."""
    candidates = []
    
    # 1. Two parallel lines of 15 points each
    pts_2line = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(pts_2line)
    
    # 2. Three parallel lines of 10 points each
    pts_3line = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(pts_3line)
    
    # 3. Four parallel lines
    pts_4line = [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + [(x, 2) for x in range(7)] + [(x, 3) for x in range(7)]
    candidates.append(pts_4line)
    
    # 4. 6x5 Grid
    pts_grid = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(pts_grid)
    
    # 5. 10x3 Grid
    pts_grid2 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(pts_grid2)
    
    # 6. 5x6 Grid (transposed)
    pts_grid3 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(pts_grid3)
    
    # 7. Concentrated on few y-values with structured x
    # Try y in {0, 2, 4} with 10 points each, x spread
    pts_spread = []
    for y in [0, 2, 4]:
        for x in range(10):
            pts_spread.append((x, y))
    candidates.append(pts_spread)
    
    # 8. Try to create many triangles with area=2
    # Points on lattice where differences give cross products of 2
    # e.g., points where coordinates are even
    pts_even = [(2*x, 2*y) for x in range(5) for y in range(3)]
    if len(pts_even) == 30:
        candidates.append(pts_even)
    
    # 9. Try modular structure: points on a torus-like pattern
    # Use points (x, y) where x mod something and y mod something
    pts_mod = []
    for i in range(30):
        x = i % 10
        y = (i // 10) * 2
        pts_mod.append((x, y))
    candidates.append(pts_mod)
    
    # 10. Try to maximize area=1 multiplicity
    # Adjacent grid points create many unit-area triangles
    pts_unit = [(x, y) for x in range(6) for y in range(5)]
    # Shift to center
    pts_unit_centered = [(x-2, y-2) for x, y in pts_unit]
    candidates.append(pts_unit_centered)
    
    # 11. Try a 15x2 grid centered
    pts_15x2 = [(x, y) for x in range(15) for y in range(2)]
    candidates.append(pts_15x2)
    
    # 12. Try points on two close parallel lines with offset
    pts_offset = [(x, 0) for x in range(15)] + [(x+1, 1) for x in range(15)]
    candidates.append(pts_offset)
    
    # 13. Try a 30-point configuration on a circle (approximate)
    pts_circle = []
    for i in range(30):
        angle = 2 * 3.14159265 * i / 30
        x = round(20 * np.cos(angle))
        y = round(20 * np.sin(angle))
        pts_circle.append((x, y))
    # Deduplicate
    seen = set()
    unique_circle = []
    for p in pts_circle:
        if p not in seen:
            seen.add(p)
            unique_circle.append(p)
    # Pad if needed
    if len(unique_circle) < 30:
        for x in range(30):
            for y in range(30):
                if (x, y) not in seen and len(unique_circle) < 30:
                    unique_circle.append((x, y))
                    seen.add((x, y))
    candidates.append(unique_circle[:30])
    
    return candidates

def _local_search(points, current_score, deadline):
    """Local search to improve the score."""
    current_points = [p for p in points]
    best_score = current_score
    best_points = [p for p in points]
    no_improve_count = 0
    max_no_improve = 100
    
    t_start = time.time()
    
    while time.time() < deadline:
        improved = False
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            x, y = current_points[idx]
            # Try various moves
            moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1),
                     (2, 0), (-2, 0), (0, 2), (0, -2),
                     (2, 1), (-2, -1), (2, -1), (-2, 1),
                     (1, 2), (-1, -2), (1, -2), (-1, 2)]
            
            for dx, dy in moves:
                if time.time() > deadline:
                    break
                    
                new_x, new_y = x + dx, y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pt = (new_x, new_y)
                # Check if new_pt is already in current_points (excluding idx)
                if new_pt in current_points[:idx] + current_points[idx+1:]:
                    continue
                
                # Apply move
                old_pt = current_points[idx]
                current_points[idx] = new_pt
                
                s = _score_np(current_points)
                
                if s > best_score:
                    best_score = s
                    best_points = [p for p in current_points]
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    current_points[idx] = old_pt
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
        
        if not improved:
            # Try a random restart from best
            if random.random() < 0.3:
                current_points = [p for p in best_points]
                no_improve_count = 0
            else:
                break
        
    return best_points, best_score

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    candidates = _structured_candidates()
    
    # Evaluate candidates
    best_points = None
    best_score = 0
    
    for pts in candidates:
        if not _valid(pts):
            continue
        
        s = _score_np(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        # Fallback
        return [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Local Search
    current_points, current_score = _local_search(best_points, best_score, DEADLINE)
    
    return current_points
