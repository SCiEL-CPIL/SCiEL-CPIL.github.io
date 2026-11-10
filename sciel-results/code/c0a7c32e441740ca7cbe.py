"""Combined approach: Fast vectorized scoring + diverse candidates + simulated annealing."""

import time
import random
import math
import numpy as np
from collections import Counter

def _doubled_area_det(p1, p2, p3):
    return abs(
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    )

def _max_multiplicity_fast(points):
    """
    Vectorized computation of max multiplicity of positive doubled area.
    points: list of 30 (x, y) tuples or a numpy array of shape (30, 2).
    """
    if isinstance(points, list):
        pts = np.array(points, dtype=np.int32)
    else:
        pts = points
    
    n = len(pts)
    if n < 3:
        return 0
    
    # We need to compute determinants for all triples (i, j, k) with i < j < k.
    # det = | (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k) |
    
    # To vectorize, we can generate all unique triples.
    # Number of triples for n=30 is 30*29*28/6 = 4060. This is small enough to handle.
    
    # Generate indices
    idx = np.triu_indices(n, k=2)
    # idx[0] are i's, idx[1] are j's, but we need k > j.
    # np.triu_indices(n, k=2) gives (i, j) with i < j. We need (i, j, k) with i < j < k.
    # It's easier to just use itertools or a small loop to build the array of triples,
    # then vectorize the determinant calculation.
    
    # Let's build the array of triples first.
    # For n=30, this is fast.
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    
    if not triples:
        return 0
        
    triples_arr = np.array(triples, dtype=np.int32)
    i_idx = triples_arr[:, 0]
    j_idx = triples_arr[:, 1]
    k_idx = triples_arr[:, 2]
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    # det = (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k)
    dx_ik = pi[:, 0] - pk[:, 0]
    dy_ik = pi[:, 1] - pk[:, 1]
    dx_jk = pj[:, 0] - pk[:, 0]
    dy_jk = pj[:, 1] - pk[:, 1]
    
    dets = np.abs(dx_ik * dy_jk - dy_ik * dx_jk)
    
    # Filter out zero areas
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
        
    # Count frequencies
    # Using np.unique is fast
    _, counts = np.unique(dets, return_counts=True)
    return int(np.max(counts))

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Rectangular Grids
    # 5x6, 6x5, 4x8, 8x4, 3x10, 10x3, 2x15, 15x2
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            # Center it
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape
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
    if len(cand_diamond) == 30:
        candidates.append(cand_diamond)

    # 3. Circle Shape
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
    if len(cand_circle) == 30:
        candidates.append(cand_circle)
        
    # 4. Random Clusters (more dense)
    for _ in range(5):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))
        
    # 5. Parabolic / Curved patterns?
    # y = x^2 mod something? Or just specific curves.
    # Let's try a set of points on a parabola y = x^2/4 scaled?
    # Or just random.
    
    return candidates

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Use 85s deadline to be safe.
    deadline = start_time + 85.0
    
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity_fast(best_points)
        
    # Simulated Annealing / Hill Climbing
    # We use numpy arrays for speed during the search loop.
    current_points = np.array(best_points, dtype=np.int32)
    current_score = best_score
    
    # Pre-allocate some structures if possible, but numpy array manipulation is key.
    
    # Temperature schedule
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Track global best
    global_best_points = current_points.copy()
    global_best_score = current_score
    
    steps = 0
    last_improvement_step = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx].copy()
        
        # Pick a new position
        # Try nearby positions. Radius 1 or 2.
        # To speed up, we can just pick a random neighbor within a small radius.
        radius = random.choice([1, 1, 2, 3])
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue
            
        new_p = old_p + np.array([dx, dy], dtype=np.int32)
        
        # Check bounds
        if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
            continue
            
        # Check collision
        # Since n=30, checking collision is fast.
        # We can just check if new_p is in current_points.
        # current_points is shape (30,2).
        # diff = current_points - new_p
        # collision = np.all(diff == 0, axis=1).any()
        
        # Optimization: We only care if new_p equals any point except the one we are moving.
        # But checking all 30 is fine.
        diff = current_points - new_p
        if np.any(np.all(diff == 0, axis=1)):
            continue
            
        # Perform move
        current_points[idx] = new_p
        
        # Evaluate
        new_score = _max_multiplicity_fast(current_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        
        if delta > 0:
            # Always accept improvement
            current_score = new_score
            if new_score > global_best_score:
                global_best_score = new_score
                global_best_points = current_points.copy()
                last_improvement_step = steps
        else:
            # Metropolis acceptance
            # Since scores are integers, delta is integer.
            # T is float.
            # If T is low, only accept if delta is small negative or zero?
            # Actually, standard SA: accept with prob exp(delta/T)
            if delta == 0:
                # Accept neutral moves to explore
                current_score = new_score
            else:
                # delta < 0
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        current_points[idx] = old_p
                else:
                    # Revert
                    current_points[idx] = old_p
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: Restart if stuck for too long?
        # If steps - last_improvement_step > 1000, maybe reset T or randomize?
        # For now, let SA handle it.
        
    return global_best_points.tolist()

# Helper to ensure we don't exceed time in _max_multiplicity_fast if it's slow
# But it's vectorized and n=30 is tiny.

# Quick sanity check / test if run directly
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")
    print(f"Score: {_max_multiplicity_fast(pts)}")
    print(f"Points: {pts}")
