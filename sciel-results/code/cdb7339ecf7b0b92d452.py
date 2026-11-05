import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety
    
    # Helper: Check if 4 points are collinear using cross product
    def are_collinear(p1, p2, p3, p4):
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        return cross(p1, p2, p3) == 0 and cross(p1, p2, p4) == 0

    def is_valid(points):
        # No 4 collinear - check all combinations of 4
        n = len(points)
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        if are_collinear(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Helper: Compute Score (Largest subset with no 3 collinear) using bitmask B&B
    def calc_score(points):
        n = len(points)
        
        # Precompute collinear triples using NumPy for speed
        # triple_mask[i] is a bitmask of all j such that i,j,k are collinear for some k
        # Actually, we need: for each pair (i,j), the set of k such that i,j,k are collinear
        # But for B&B, we need: for each vertex v, which pairs (u,w) form a triple with v
        
        # Let's precompute: for each vertex v, a set of edges (u,w) where u<w and v,u,w are collinear
        # When we add v to chosen, we cannot have both u and w in chosen for any such edge
        
        # Alternative: use a different B&B formulation
        # Sort vertices by "difficulty" (number of collinear triples they're in)
        
        # First, count how many triples each vertex is in
        triple_count = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                        triple_count[i] += 1
                        triple_count[j] += 1
                        triple_count[k] += 1
        
        # Sort vertices by triple count (descending) - hard vertices first
        order = sorted(range(n), key=lambda v: -triple_count[v])
        
        # Precompute for each position in order, the list of (pos_u, pos_w) pairs that form a triple
        # where pos_u < pos_w in the ordering
        pairs_for_pos = [[] for _ in range(n)]
        for i_idx in range(n):
            for j_idx in range(i_idx+1, n):
                for k_idx in range(j_idx+1, n):
                    i, j, k = order[i_idx], order[j_idx], order[k_idx]
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                        # Triple found: i, j, k
                        # For vertex i (at position i_idx), pair (j, k) is forbidden
                        # For vertex j (at position j_idx), pair (i, k) is forbidden
                        # For vertex k (at position k_idx), pair (i, j) is forbidden
                        pairs_for_pos[i_idx].append((j_idx, k_idx))
                        pairs_for_pos[j_idx].append((i_idx, k_idx))
                        pairs_for_pos[k_idx].append((i_idx, j_idx))
        
        best_size = 0
        chosen = [False] * n
        
        def backtrack(pos, current_size):
            nonlocal best_size
            if pos == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pruning: even if we take all remaining, can't beat best
            if current_size + (n - pos) <= best_size:
                return
            
            # Option 1: Skip this vertex
            backtrack(pos + 1, current_size)
            
            # Option 2: Include this vertex
            v_pos = pos
            valid = True
            for u_pos, w_pos in pairs_for_pos[v_pos]:
                if chosen[u_pos] and chosen[w_pos]:
                    valid = False
                    break
            
            if valid:
                chosen[v_pos] = True
                backtrack(pos + 1, current_size + 1)
                chosen[v_pos] = False
        
        backtrack(0, 0)
        return best_size

    def generate_structured():
        """Generate points using structured patterns that tend to have lower scores"""
        patterns = []
        
        # Pattern 1: Points on a circle with slight perturbations
        for angle_offset in [0, 0.1, 0.2, 0.3]:
            pts = []
            for i in range(20):
                angle = 2 * np.pi * i / 20 + angle_offset
                r = 50 + 10 * np.sin(i * 1.7)
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            # Remove duplicates
            unique_pts = list(dict.fromkeys(pts))
            while len(unique_pts) < 20:
                unique_pts.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique_pts[:20])
        
        # Pattern 2: Grid with perturbations
        for offset in [0, 1, 2]:
            pts = []
            for i in range(4):
                for j in range(5):
                    x = -40 + i * 20 + offset + random.randint(-3, 3)
                    y = -40 + j * 20 + offset + random.randint(-3, 3)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            unique_pts = list(dict.fromkeys(pts))
            while len(unique_pts) < 20:
                unique_pts.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique_pts[:20])
        
        # Pattern 3: Random but clustered
        for center in [(0, 0), (20, 20), (-20, -20)]:
            pts = []
            for i in range(20):
                cx, cy = center
                x = cx + random.randint(-30, 30)
                y = cy + random.randint(-30, 30)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique_pts = list(dict.fromkeys(pts))
            while len(unique_pts) < 20:
                unique_pts.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique_pts[:20])
        
        return patterns

    def generate_random():
        points = []
        while len(points) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in points:
                points.append(p)
        return points

    best_points = None
    best_score = 21

    # Try structured patterns first
    structured_patterns = generate_structured()
    for pts in structured_patterns:
        if time.time() > deadline:
            break
        if is_valid(pts):
            score = calc_score(pts)
            if score < best_score:
                best_score = score
                best_points = pts[:]

    # Random search with local improvement
    while time.time() < deadline:
        if best_points is None:
            # Generate random
            points = generate_random()
        else:
            # Local search: try to improve best_points
            points = best_points[:]
            num_moves = random.randint(1, 3)
            for _ in range(num_moves):
                idx = random.randint(0, 19)
                old_p = points[idx]
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p in points:
                    continue
                points[idx] = new_p
            
            # Also try swap moves
            if random.random() < 0.3:
                i, j = random.sample(range(20), 2)
                points[i], points[j] = points[j], points[i]
        
        if not is_valid(points):
            continue
        
        score = calc_score(points)
        
        if score < best_score:
            best_score = score
            best_points = points[:]
        
        # If we have a good score, do more local search
        if best_points is not None and best_score <= 10:
            # Intensive local search
            for _ in range(50):
                if time.time() > deadline:
                    break
                trial = best_points[:]
                idx = random.randint(0, 19)
                old_p = trial[idx]
                # Try nearby points
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_p = (old_p[0] + dx, old_p[1] + dy)
                new_p = (max(-100, min(100, new_p[0])), max(-100, min(100, new_p[1])))
                
                if new_p in trial:
                    continue
                
                trial[idx] = new_p
                
                if is_valid(trial):
                    new_score = calc_score(trial)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = trial[:]

    if best_points is None:
        # Fallback: use a known valid configuration
        best_points = [(i*10, (i*i) % 100 - 50) for i in range(20)]
        # Ensure validity
        if not is_valid(best_points):
            best_points = [(i*5, i*3) for i in range(20)]
    
    return [[p[0], p[1]] for p in best_points]
