"""Combined approach: Structured initialization + Local Search/Hill Climbing."""

import time
import random
import math
from collections import Counter
import numpy as np

def _doubled_area_det(p1, p2, p3):
    return abs(
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    )

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    # Compute all triple determinants vectorized
    # For each i < j < k, det = |xi*yj - xj*yi + xj*yk - xk*yj + xk*yi - xi*yk|
    # Using the formula: det = |(xi - xk)*(yj - yk) - (yi - yk)*(xj - xk)|
    
    # Efficient approach: compute all pairwise cross products
    # For triple (i,j,k): det = (pi - pk) x (pj - pk)
    # = pi x pj - pi x pk - pk x pj
    # where a x b = ax*by - ay*bx
    
    # Let's use the direct formula with broadcasting
    # This creates an n x n x n array which is 27000 elements for n=30, manageable
    
    # Alternative: iterate over k, compute areas for all pairs (i,j) with i<j<k
    areas = []
    for k in range(2, n):
        pk = pts[k]
        # For all i < j < k
        # det(i,j,k) = (pi - pk) x (pj - pk)
        # = (pi_x - pk_x)*(pj_y - pk_y) - (pi_y - pk_y)*(pj_j - pk_x)
        pi = pts[:k]  # points 0..k-1
        # Compute cross product of (pi - pk) and (pj - pk) for all i < j
        # pi_x - pk_x for all i in [0,k-1]
        dx_i = pi[:, 0] - pk[0]
        dy_i = pi[:, 1] - pk[1]
        # For each j in [1, k-1], compute with all i < j
        for j in range(1, k):
            dx_j = pts[j, 0] - pk[0]
            dy_j = pts[j, 1] - pk[1]
            # Cross products for all i < j
            cross = dx_i[:j] * dy_j - dy_i[:j] * dx_j
            abs_cross = np.abs(cross)
            # Filter out zeros
            nonzero = abs_cross[abs_cross > 0]
            if len(nonzero) > 0:
                areas.append(nonzero)
    
    if not areas:
        return 0
    
    all_areas = np.concatenate(areas)
    if len(all_areas) == 0:
        return 0
    
    # Count multiplicities
    vals, counts = np.unique(all_areas, return_counts=True)
    return int(counts.max())

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_diamond = []
    for x, y, d in diamond_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_diamond.append((x, y))
            if len(cand_diamond) == 30:
                break
    candidates.append(cand_diamond)

    # 3. Circle Shape (Euclidean distance)
    circle_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            circle_pts.append((x, y, x*x + y*y))
    circle_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_circle = []
    for x, y, d in circle_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_circle.append((x, y))
            if len(cand_circle) == 30:
                break
    candidates.append(cand_circle)
    
    # 4. Lattice patterns with different bases
    # Hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 2 + (row % 2)
            y = row
            hex_pts.append((x, y))
    if len(hex_pts) == 30:
        cx = 4
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # Triangular lattice
    tri_pts = []
    for row in range(6):
        for col in range(5):
            x = col
            y = row
            tri_pts.append((x, y))
    if len(tri_pts) == 30:
        cx = 2
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 6. More structured random: pick from small grid with some randomness
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit
    
    # 1. Initialize
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        
    # 2. Local Search (Hill Climbing with Random Restarts)
    current_points = best_points[:]
    current_score = best_score
    
    # Pre-calculate set of occupied points for fast collision check
    occupied = set(current_points)
    
    # Track steps without improvement for restart logic
    steps_since_improvement = 0
    max_steps_without_improvement = 200
    
    while time.time() < deadline:
        # If stuck, do a random restart from best
        if steps_since_improvement > max_steps_without_improvement:
            current_points = best_points[:]
            current_score = best_score
            occupied = set(current_points)
            steps_since_improvement = 0
            max_steps_without_improvement = random.randint(100, 300)
            continue
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Pick a new position - try nearby positions
        moved = False
        # Randomly choose between small moves and larger jumps
        if random.random() < 0.7:
            max_radius = random.choice([1, 2, 3])
        else:
            max_radius = random.choice([3, 5, 8])
        
        for radius in range(1, max_radius + 1):
            if moved:
                break
            # Try random offsets within this radius
            for _ in range(3):  # Try a few random offsets per radius
                dx = random.randint(-radius, radius)
                dy = random.randint(-radius, radius)
                if dx == 0 and dy == 0:
                    continue
                new_p = (old_p[0] + dx, old_p[1] + dy)
                # Check bounds
                if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30:
                    # Check collision
                    if new_p not in occupied:
                        # Perform move
                        occupied.remove(old_p)
                        occupied.add(new_p)
                        current_points[idx] = new_p
                        
                        # Evaluate
                        new_score = _max_multiplicity(current_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            steps_since_improvement = 0
                            if new_score > best_score:
                                best_score = new_score
                                best_points = current_points[:]
                        else:
                            # Revert move
                            occupied.remove(new_p)
                            occupied.add(old_p)
                            current_points[idx] = old_p
                            steps_since_improvement += 1
                        moved = True
                        break
            if moved:
                break
        if not moved:
            steps_since_improvement += 1
        
    return best_points
