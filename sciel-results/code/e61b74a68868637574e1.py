import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin for safety
    
    # Grid initialization
    # Try to create a set with high collinearity but no 4-collinear
    # Start with a 4x5 grid and perturb
    points = []
    for i in range(4):
        for j in range(5):
            points.append([i, j])
    
    # Randomize slightly to break 4-collinearities
    # We will use local search to fix validity and minimize score
    
    def check_no_four_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        # Check if collinear
                        # (x2-x1)(y3-y1) == (x3-x1)(y2-y1)
                        # and (x2-x1)(y4-y1) == (x4-x1)(y2-y1)
                        if ((p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]) and
                            (p2[0]-p1[0])*(p4[1]-p1[1]) == (p4[0]-p1[0])*(p2[1]-p1[1])):
                            return False
        return True

    def get_max_no_three_collinear(pts):
        n = len(pts)
        # Precompute collinearity
        # collinear[p][q] is a set of r such that p,q,r are collinear
        # Actually, for the search, we need to know if adding p creates a triple with existing points.
        # Let's precompute for each pair (i,j), the list of k such that i,j,k are collinear.
        collinear_pairs = {}
        for i in range(n):
            for j in range(i+1, n):
                collinear_pairs[(i,j)] = []
                for k in range(n):
                    if k == i or k == j: continue
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                        collinear_pairs[(i,j)].append(k)
        
        best = 0
        # Branch and bound
        # State: current subset (list of indices), next index to consider
        # We can use a recursive function
        def branch_and_bound(current_subset, next_idx):
            nonlocal best
            if len(current_subset) + (n - next_idx) <= best:
                return
            
            # Check if we can add any more points
            # Actually, we iterate through points
            # To speed up, we can precompute conflicts
            
            if next_idx == n:
                if len(current_subset) > best:
                    best = len(current_subset)
                return
            
            # Option 1: Skip point next_idx
            branch_and_bound(current_subset, next_idx + 1)
            
            # Option 2: Add point next_idx
            # Check if it forms a triple with any two points in current_subset
            valid = True
            for i in current_subset:
                for j in current_subset:
                    if i == j: continue
                    pair = (i, j) if i < j else (j, i)
                    if next_idx in collinear_pairs.get(pair, []):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                branch_and_bound(current_subset + [next_idx], next_idx + 1)
                
        branch_and_bound([], 0)
        return best

    best_points = points
    best_score = 20 # Max possible
    
    # First, ensure validity. If not valid, randomize until valid.
    # The grid is invalid. We need to find a valid starting point.
    # Let's try random perturbations.
    for _ in range(1000):
        if time.time() > deadline: break
        # Randomize points in [-10, 10]
        cand = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(20)]
        if len(set(map(tuple, cand))) < 20: continue
        if check_no_four_collinear(cand):
            best_points = cand
            best_score = get_max_no_three_collinear(cand)
            break
            
    if best_score == 20:
        # If we didn't find a valid one, force one by construction?
        # Or just return the grid (invalid) -> Score 0.
        # We must return a valid one.
        # Let's try a known valid configuration:
        # Points on a circle? No, 3 can be collinear.
        # Let's try a convex position set (no 3 collinear). Score 20.
        # But we want lower.
        # If we can't find a valid one quickly, we return a convex set.
        best_points = [[x, x*x] for x in range(-10, 10)]
        best_score = 20

    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = best_points[idx]
        # Try nearby positions
        found_improvement = False
        for _ in range(20):
            new_p = [old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5)]
            if abs(new_p[0]) > 100 or abs(new_p[1]) > 100:
                continue
            if new_p in best_points:
                continue
            
            cand = best_points.copy()
            cand[idx] = new_p
            
            if not check_no_four_collinear(cand):
                continue
            
            score = get_max_no_three_collinear(cand)
            if score < best_score:
                best_score = score
                best_points = cand
                found_improvement = True
                break
            elif score == best_score and random.random() < 0.1:
                # Allow sideways moves to escape local minima
                best_points = cand
                break
        
    return best_points
