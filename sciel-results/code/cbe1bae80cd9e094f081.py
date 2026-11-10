"""Combined approach: Structured initialization + Local Search/Hill Climbing."""

import time
import random
import math
from collections import Counter

def _doubled_area_det(p1, p2, p3):
    return abs(
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    )

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    counter = Counter()
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                d = _doubled_area_det(pi, pj, pk)
                if d > 0:
                    counter[d] += 1
    if not counter:
        return 0
    return max(counter.values())

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    # We know 5x6 and 6x5 are strong. 4x8, 8x4, 3x10, 10x3 are also candidates.
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            # Center it to be roughly in [-30, 30] (though translation doesn't matter for score, 
            # it's good practice for the local search to have room to move)
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    # Generate points sorted by |x| + |y|
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    # Take first 30 unique points
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
    
    # 4. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
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
    # Since evaluation is fast (~4000 triples), we can do many steps.
    # We use a simple hill climbing approach: try to improve the current best.
    # We also allow "restarts" from the best solution found so far if stuck, 
    # or just keep perturbing the global best.
    
    # To be safe and effective, we will perform a series of perturbation steps 
    # on the current best solution. If we find a better one, we update.
    # We will also occasionally try to "jump" by randomizing a few points.
    
    current_points = best_points[:]
    current_score = best_score
    
    # Pre-calculate set of occupied points for fast collision check
    occupied = set(current_points)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Pick a new position
        # Try nearby positions first (radius 1, then 2, etc.)
        moved = False
        for radius in [1, 2, 3]:
            if moved:
                break
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
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
                                if new_score > best_score:
                                    best_score = new_score
                                    best_points = current_points[:]
                            else:
                                # Revert move
                                occupied.remove(new_p)
                                occupied.add(old_p)
                                current_points[idx] = old_p
                            moved = True
                            break
                if moved:
                    break
            
        # If we've done many steps without improving the global best, 
        # we might be stuck in a local optimum. 
        # Simple strategy: If no global improvement in last 500 steps, 
        # restart local search from the global best with a random perturbation.
        # For simplicity, we just keep doing random moves on the current local best.
        # Since we track global best, we are safe.
        
    return best_points
