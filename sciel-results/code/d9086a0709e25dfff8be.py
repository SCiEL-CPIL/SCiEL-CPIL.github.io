"""Combined approach: NumPy-vectorized scoring + diverse initial candidates + simulated annealing."""

import time
import random
import numpy as np
from collections import Counter

def _score_numpy(points):
    """Compute max multiplicity of positive doubled area using NumPy vectorization."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all triples determinants efficiently
    # For each pair (i,j), compute vector (pj-pi), then for each k, cross with (pk-pi)
    # Better: use the formula det = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Vectorized: for all i<j<k
    # Alternative: compute all pairwise determinants with a fixed point as origin
    
    # Efficient approach: for each point as "origin", compute determinants with all pairs of others
    # det(pi, pj, pk) = (pj - pi) x (pk - pi)
    # = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    
    # Let's compute all n*(n-1)*(n-2)/6 determinants
    # Use broadcasting: for each i, compute vectors from pi to all pj (j != i)
    # Then for each pair of those vectors, compute cross product
    
    all_dets = []
    for i in range(n):
        pi = pts[i]
        # Vectors from pi to all other points
        vecs = pts - pi  # shape (n, 2)
        # We need cross products of vecs[j] and vecs[k] for j < k, j,k != i
        # Actually j and k can be any index != i, but we want unique triples
        # Let's just compute all cross products for pairs in vecs where both indices != i
        # and then filter for unique triples
        
        # Simpler: compute all cross products vecs[a] x vecs[b] for a < b
        # But we need to ensure i, a, b are all distinct and form a unique triple
        # Since vecs[i] = 0, cross with 0 is 0, so we can ignore those
        
        # Compute all pairwise cross products
        # vecs has shape (n, 2)
        # cross(a, b) = a[0]*b[1] - a[1]*b[0]
        # For all a < b:
        x = vecs[:, 0]
        y = vecs[:, 1]
        # Cross product matrix: C[a,b] = x[a]*y[b] - y[a]*x[b]
        # We only need upper triangle (a < b) and a != i, b != i
        # But vecs[i] = 0, so C[i, b] = 0 and C[a, i] = 0 automatically
        
        # Compute full cross product matrix
        C = np.outer(x, y) - np.outer(y, x)
        # Get upper triangle indices
        rows, cols = np.triu_indices(n, k=1)
        dets = np.abs(C[rows, cols])
        # Filter: exclude triples where any index is i (but those have det=0 anyway since vecs[i]=0)
        # Also exclude zero-area
        # But we need to ensure we're counting each triple exactly once
        # Triple (i, a, b) where a > b? No, we use a < b from triu
        # The triple is (i, a, b) with i fixed, a < b, a != i, b != i
        # But if a == i or b == i, det = 0, so they won't contribute to positive areas
        # However, we might be double-counting? No, each triple (min, mid, max) is counted once
        # when i = min index? No, i is the first point in the triple in our loop
        # Actually this approach counts each triple multiple times!
        # Triple (a, b, c) is counted when i=a (with b,c), i=b (with a,c), i=c (with a,b)
        # So we'd triple-count. Let's use a different approach.
        
        # Better: just iterate over i < j < k using numpy where possible
        # For 30 points, n=30, number of triples = 4060
        # Let's just do it with a loop over i and vectorize over j,k
        pass
    
    # Let me redo this more carefully
    # For 30 points, 4060 triples. Even a pure Python loop might be okay, but let's vectorize.
    
    # Approach: create arrays of all triple indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_arr = np.array(i_idx, dtype=np.int64)
    j_arr = np.array(j_idx, dtype=np.int64)
    k_arr = np.array(k_idx, dtype=np.int64)
    
    pi = pts[i_arr]
    pj = pts[j_arr]
    pk = pts[k_arr]
    
    # det = |(pj - pi) x (pk - pi)|
    # = |(pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)|
    det = np.abs((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - 
                 (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
    
    # Exclude zero-area
    positive_dets = det[det > 0]
    if len(positive_dets) == 0:
        return 0
    
    # Count multiplicities
    # Use np.unique with return_counts
    unique_dets, counts = np.unique(positive_dets, return_counts=True)
    return int(counts.max())

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    return _score_numpy(points)

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]
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
    
    # 4. Diagonal patterns
    for offset in range(-5, 6):
        pts = []
        for i in range(30):
            x = i % 6
            y = i // 6 + offset
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
            else:
                pts.append((x, 0))
        # Ensure distinctness
        unique_pts = []
        seen = set()
        for p in pts:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
            if len(unique_pts) == 30:
                break
        if len(unique_pts) == 30:
            candidates.append(unique_pts)
    
    # 5. Parabolic patterns
    for a in [1, -1, 2, -2]:
        pts = []
        for i in range(-5, 5):
            x = i
            y = a * i * i // 4
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        # Need 30 points, so add more
        if len(pts) < 30:
            for i in range(-10, 10):
                x = i
                y = a * i * i // 6
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in set(pts):
                    pts.append((x, y))
                if len(pts) == 30:
                    break
        if len(pts) == 30:
            candidates.append(pts)
    
    # 6. Random Clusters
    for _ in range(15):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 7. Hexagonal-ish patterns
    for _ in range(5):
        pts = []
        for i in range(6):
            for j in range(5):
                x = i + (j % 2) * 0  # offset for hex
                y = j
                pts.append((x, y))
        # Adjust to get 30 distinct
        unique_pts = []
        seen = set()
        for p in pts:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
            if len(unique_pts) == 30:
                break
        if len(unique_pts) == 30:
            candidates.append(unique_pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    # 1. Generate and evaluate initial candidates
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        
    # 2. Simulated Annealing
    current_points = best_points[:]
    current_score = best_score
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    T = T_start
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Pick a new position with some randomness
        # Use a Gaussian-like perturbation
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx = random.choice([-1, 1])
        
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        # Check bounds
        if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
            T *= cooling_rate
            continue
        
        # Check collision
        if new_p in current_points:
            T *= cooling_rate
            continue
        
        # Perform move
        current_points[idx] = new_p
        new_score = _max_multiplicity(current_points)
        delta_score = new_score - current_score
        
        # Accept or reject
        if delta_score > 0 or random.random() < np.exp(delta_score / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_points[idx] = old_p
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # If stuck for too long, try a bigger jump
        if no_improve_count > 100:
            no_improve_count = 0
            # Randomly perturb 3-5 points
            for _ in range(random.randint(3, 5)):
                idx2 = random.randint(0, 29)
                old_p2 = current_points[idx2]
                dx2 = random.randint(-5, 5)
                dy2 = random.randint(-5, 5)
                new_p2 = (old_p2[0] + dx2, old_p2[1] + dy2)
                if -30 <= new_p2[0] <= 30 and -30 <= new_p2[1] <= 30 and new_p2 not in current_points:
                    current_points[idx2] = new_p2
            current_score = _max_multiplicity(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
    
    return best_points
