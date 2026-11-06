import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin for safety
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Check if any 4 points are collinear in a list of points
    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Check if there is a 4th point on the line defined by i, j, k
                    for m in range(k + 1, n):
                        if are_collinear(points[i], points[j], points[k]) and are_collinear(points[i], points[j], points[m]):
                            return True
        return False

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    # Uses bitmasks for speed
    def max_no_3_collinear(points):
        n = len(points)
        
        # Precompute collinearity masks
        # mask[i] is a bitmask where bit j is set if points i and j are collinear with some third point?
        # No, we need to know if adding i to S creates a triple.
        # A triple (i, j, k) is bad if i, j, k are collinear.
        # Let's precompute for each pair (i, j), the bitmask of k such that i, j, k are collinear.
        # Or simpler: for each i, and each other j, count how many k are collinear with i and j?
        
        # Better: Precompute for each i, a list of masks.
        # Actually, let's precompute a 2D array of masks? No, n=20 is small.
        # Let's precompute for each i, the set of pairs (j, k) that are collinear with i.
        # Then, when checking if we can add i to S, we check if any pair in S is in bad_pairs[i].
        
        bad_pairs_for = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        bad_pairs_for[i].add((j, k))
                        bad_pairs_for[j].add((i, k))
                        bad_pairs_for[k].add((i, j))
        
        best = 0
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')
            # Prune
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip point start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include point start_idx
            # Check if adding start_idx to S is valid
            # It is valid if there is no pair (j, k) in S such that (j, k) is in bad_pairs_for[start_idx]
            valid = True
            for j in range(start_idx):
                if not (S_mask & (1 << j)):
                    continue
                for k in range(j + 1, start_idx):
                    if not (S_mask & (1 << k)):
                        continue
                    if (j, k) in bad_pairs_for[start_idx] or (k, j) in bad_pairs_for[start_idx]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)
                
        backtrack(0, 0)
        return best

    # Strategy: Greedy Construction
    # We want to maximize the number of collinear triples to minimize the max independent set (no-3-collinear).
    # Greedy: Place points one by one.
    # At each step, consider all candidate points (x, y) in [-100, 100] x [-100, 100] that are distinct and don't create 4-collinear.
    # Score a candidate by how many new collinear triples it creates.
    # Pick the candidate with the highest score.
    
    # To make this efficient, we don't check all 200x200 points at each step.
    # We can restrict candidates to points that are collinear with at least 2 existing points (to form a triple) 
    # or just random sampling if none exist.
    # Actually, a good heuristic is to try to extend existing lines.
    
    def greedy_construct():
        pts = []
        # Start with a few random points to break symmetry
        for _ in range(3):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    pts.append([x, y])
                    break
        
        while len(pts) < 20:
            if time.time() > deadline - 5:
                break
                
            best_candidate = None
            best_score = -1
            
            # Generate candidates
            # 1. Points that are collinear with pairs of existing points
            candidates = set()
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    p1 = pts[i]
                    p2 = pts[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Find integer points on this line within bounds
                    # We can just sample a few points along the line or check specific offsets
                    # To keep it simple and fast, let's just check a limited set of "nearby" points
                    # Or better: iterate over all possible x in [-100, 100] and see if y is integer?
                    # That's 200 checks per pair. 190 pairs * 200 = 38000 checks. Fast.
                    
                    # Let's just add the endpoints and a few midpoints? No, we want to maximize triples.
                    # Any point on the line is a candidate.
                    # We'll add all valid integer points on the line defined by p1, p2 within bounds.
                    # But there might be many.
                    # Let's just sample a few random points on the line?
                    # Or just check if p1 and p2 define a line with small slope/intercept?
                    
                    # For simplicity and speed, let's just add p1, p2 neighbors?
                    # No, that's weak.
                    
                    # Let's iterate x from -100 to 100.
                    # If dx != 0:
                    #   y = p1[1] + dy/dx * (x - p1[0])
                    #   Check if y is integer.
                    # If dx == 0:
                    #   x = p1[0], y varies.
                    
                    if dx == 0:
                        for y in range(-100, 101):
                            p = [p1[0], y]
                            if p not in pts:
                                candidates.add((p[0], p[1]))
                    else:
                        # y = p1[1] + (dy/dx) * (x - p1[0])
                        # dy * (x - p1[0]) must be divisible by dx
                        for x in range(-100, 101):
                            num = dy * (x - p1[0])
                            if num % dx == 0:
                                y = p1[1] + num // dx
                                if -100 <= y <= 100:
                                    p = [x, y]
                                    if p not in pts:
                                        candidates.add((p[0], p[1]))
            
            # Also add some random points to escape local minima
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    candidates.add((x, y))
            
            # Evaluate candidates
            for cx, cy in candidates:
                p = [cx, cy]
                new_pts = pts + [p]
                # Check 4-collinear
                if has_4_collinear(new_pts):
                    continue
                
                # Count new triples
                # A new triple must include p.
                # So count pairs (i, j) in pts such that p, pts[i], pts[j] are collinear.
                triples = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if are_collinear(p, pts[i], pts[j]):
                            triples += 1
                
                if triples > best_score:
                    best_score = triples
                    best_candidate = p
            
            if best_candidate is None:
                # Fallback: pick a random valid point
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p not in pts and not has_4_collinear(pts + [p]):
                        best_candidate = p
                        break
                # If still None, just add a random point and hope for the best (shouldn't happen)
                if best_candidate is None:
                    best_candidate = [random.randint(-100, 100), random.randint(-100, 100)]
            
            pts.append(best_candidate)
            
        return pts

    # Main loop
    best_config = None
    best_score = 21  # Max possible score is 20
    
    # Run greedy construction a few times
    for _ in range(10):
        if time.time() > deadline - 10:
            break
            
        pts = greedy_construct()
        
        # Validate
        if len(set(map(tuple, pts))) != 20:
            continue
        if has_4_collinear(pts):
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
        # If we found a good score, try to improve it with local search
        if best_score <= 12 and time.time() < deadline - 5:
            # Local search: move a random point to a new location if it improves the score
            improved = [list(p) for p in best_config]
            for step in range(50):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, 19)
                old_p = improved[idx][:]
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                improved[idx] = [new_x, new_y]
                
                if len(set(map(tuple, improved))) != 20 or has_4_collinear(improved):
                    improved[idx] = old_p
                    continue
                    
                new_score = max_no_3_collinear(improved)
                if new_score < best_score:
                    best_score = new_score
                    best_config = improved
                else:
                    improved[idx] = old_p
                    
    if best_config is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config
