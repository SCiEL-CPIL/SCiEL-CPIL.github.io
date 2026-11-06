import time
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
            line_counts = defaultdict(int)
            for j in range(n):
                if i == j:
                    continue
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                if dx == 0 and dy == 0:
                    continue
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                g = abs(dx)
                h = abs(dy)
                while g != 0 and h != 0:
                    g, h = h, g % h
                g = g if g != 0 else 1
                dx, dy = dx // g, dy // g
                line_counts[(dx, dy)] += 1
            if any(c >= 3 for c in line_counts.values()):
                return True
        return False

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    def max_no_3_collinear(points):
        n = len(points)
        
        # Precompute for each i, the set of pairs (j, k) that are collinear with i
        # Actually, for the branch and bound, we need to check if adding point i
        # creates a collinear triple with any two points already in the set.
        
        # Let's use a simpler bitmask approach with precomputed "conflict" masks.
        # conflict[i] is a bitmask of points that are collinear with i? No, that's not enough.
        # We need: if i is in S, then for any j, k in S, i,j,k must not be collinear.
        
        # Let's precompute for each pair (j, k), the set of points i that are collinear with j and k.
        # This is O(n^3) to precompute, which is fine for n=20.
        # collinear_with[j][k] = bitmask of i such that i, j, k are collinear.
        
        collinear_with = [[0] * n for _ in range(n)]
        for j in range(n):
            for k in range(j + 1, n):
                mask = 0
                for i in range(n):
                    if i != j and i != k:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << i)
                collinear_with[j][k] = mask
                collinear_with[k][j] = mask

        best = 0
        
        # Backtracking
        # S_mask: bitmask of selected points
        # last_idx: the index of the last point added (to avoid checking pairs where both < last_idx? No, we need to check all pairs in S)
        # Optimization: When adding point i, we only need to check pairs (j, k) in S such that j, k < i? 
        # No, because S can contain points with index > i if we are not iterating in order.
        # But if we iterate i from 0 to n-1, and decide to include or exclude, then when we consider i,
        # all points in S have index < i. 
        # So we only need to check pairs (j, k) where j, k < i and j, k are in S.
        
        def backtrack(S_mask, i):
            nonlocal best
            current_len = bin(S_mask).count('1')
            
            # Upper bound: current_len + (n - i)
            if current_len + (n - i) <= best:
                return
            
            if i == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Exclude i
            backtrack(S_mask, i + 1)
            
            # Option 2: Include i
            # Check if i is collinear with any pair (j, k) in S_mask where j, k < i
            valid = True
            # Iterate through all pairs in S_mask
            # Since j, k < i, we can just iterate j from 0 to i-1
            for j in range(i):
                if S_mask & (1 << j):
                    # Check if there is any k in S_mask, k < j, such that i, j, k are collinear
                    # We precomputed collinear_with[j][k]? No, collinear_with[j][k] gives points collinear with j and k.
                    # We need to know if i is in collinear_with[j][k].
                    # But we also need k to be in S_mask.
                    # So for a fixed j in S, we need to check if there exists k in S, k < j, such that i is in collinear_with[j][k].
                    
                    # Let's optimize this check.
                    # For each j in S, let M_j = { k in S, k < j | i is collinear with j and k }
                    # If M_j is not empty, then invalid.
                    
                    # We can precompute for each i, and each j, a bitmask of k < j such that i, j, k are collinear.
                    # Let's call it bad_k_mask[i][j].
                    # Then the check is: (S_mask & bad_k_mask[i][j]) != 0.
                    
                    # We didn't precompute bad_k_mask. Let's do it on the fly or precompute outside.
                    # Precomputing outside is better.
                    
                    pass # Placeholder, will use precomputed bad_k_mask in actual implementation below
                    
            # Since the on-the-fly check in the inner loop is complex, let's use the precomputed bad_k_mask approach.
            # I will re-implement the backtracking with precomputed bad_k_mask.
            
            return # This function is a stub, replaced below

        # Re-implementation of max_no_3_collinear with precomputed bad_k_mask
        bad_k_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                mask = 0
                for k in range(j):
                    if are_collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
                bad_k_mask[i][j] = mask

        def backtrack2(S_mask, i):
            nonlocal best
            current_len = bin(S_mask).count('1')
            
            if current_len + (n - i) <= best:
                return
            
            if i == n:
                if current_len > best:
                    best = current_len
                return
            
            # Exclude i
            backtrack2(S_mask, i + 1)
            
            # Include i
            valid = True
            for j in range(i):
                if S_mask & (1 << j):
                    if S_mask & bad_k_mask[i][j]:
                        valid = False
                        break
            if valid:
                backtrack2(S_mask | (1 << i), i + 1)

        backtrack2(0, 0)
        return best

    # Strategy: Generate structured templates
    # We want sets of 20 points with many 3-collinearities but no 4-collinearities.
    # Known good structures:
    # 1. Points on a parabola y = x^2 + c*x + d, but perturbed to avoid 4-collinearities?
    #    Actually, no 3 points on a parabola are collinear. So score is 20. Bad.
    # 2. We need points that ARE collinear in triples.
    #    Consider a "broken grid". 
    #    Or consider points on lines with small slopes.
    
    # Let's try a deterministic search over "affine" transformations of a base set,
    # and also some heuristic constructions.
    
    def generate_candidates():
        candidates = []
        
        # Candidate 1: Randomized local search around a "dense" initial guess
        # Initial guess: 4 lines of 5 points each? 
        # If we have 4 lines of 5 points, and they intersect nicely...
        # But we need exactly 20 distinct points.
        
        # Let's try a "Grid with holes" approach.
        # Take a 5x4 grid. 20 points.
        # In a 5x4 grid, how many 3-collinearities?
        # Rows: 4 rows of 5 -> 4 * C(5,3) = 4 * 10 = 40 triples.
        # Cols: 5 cols of 4 -> 5 * C(4,3) = 5 * 4 = 20 triples.
        # Diagonals: 
        #    Slope 1: (0,0),(1,1),(2,2),(3,3) -> 1 triple. (0,1),(1,2),(2,3) -> 1. (1,0),(2,1),(3,2) -> 1.
        #    Slope -1: similar.
        #    Other slopes: (0,0),(2,1),(4,2) etc.
        
        # A 5x4 grid has NO 4-collinearities? 
        # Wait, a row has 5 points. So 5 are collinear. That violates "no 4 collinear".
        # So we cannot use a full grid.
        
        # We need to "break" the grid so that no line has 4 points.
        # Strategy: Start with a 5x4 grid. For each line with 4 or more points, move one point.
        
        base_grid = []
        for r in range(4):
            for c in range(5):
                base_grid.append([c, r])
                
        # Check if base_grid has 4-collinear. Yes, rows and cols.
        # We need to perturb it.
        
        # Let's try a different structure: "Eisenstein" or "Gaussian" integer lattices?
        
        # Alternative: Use a known small configuration from literature if available?
        # No, let's construct.
        
        # Heuristic: 
        # 1. Start with 20 random points.
        # 2. Local search: Move a point to maximize the number of 3-collinearities, 
        #    subject to no 4-collinearities.
        
        def local_search_init():
            # Initialize with random points
            pts = []
            for _ in range(20):
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if [x, y] not in pts:
                        pts.append([x, y])
                        break
            return pts

        def count_3_collinear(pts):
            count = 0
            n = len(pts)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(pts[i], pts[j], pts[k]):
                            count += 1
            return count

        def improve(pts):
            improved = [list(p) for p in pts]
            improved_flag = True
            while improved_flag and time.time() < deadline - 10:
                improved_flag = False
                for i in range(20):
                    old_p = improved[i][:]
                    # Try moving point i
                    # To find a good new position, look at lines formed by pairs of other points.
                    # For each pair (j, k), find the line. If we place i on this line, 
                    # we might form a triple (i, j, k).
                    # But we must avoid 4-collinearities.
                    
                    best_new_p = None
                    best_score = -1
                    current_score = count_3_collinear(improved)
                    
                    # Candidate positions:
                    # 1. Points on lines defined by pairs (j, k) in improved (j != i, k != i).
                    #    Specifically, if we place i on the line through j and k, 
                    #    we form a triple. 
                    #    But we need to ensure no 4-collinear.
                    
                    candidates = set()
                    for j in range(20):
                        if j == i: continue
                        for k in range(20):
                            if k == i or k == j: continue
                            # Line through j and k
                            p1 = improved[j]
                            p2 = improved[k]
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            # Find integer points on this line within [-100, 100]
                            # We only need a few candidates to "break" 4-collinear or add triples.
                            # Actually, if we place i on the line, we add 1 triple (i,j,k).
                            # But if there are already 2 other points on this line, we get 4-collinear.
                            
                            # Let's just sample a few points on the line.
                            if dx == 0:
                                for y in range(-100, 101, 5): # Sample every 5
                                    p = [p1[0], y]
                                    if p != old_p:
                                        candidates.add(tuple(p))
                            else:
                                # Parametric: x = p1[0] + t*dx, y = p1[1] + t*dy
                                # We want integer points. 
                                # Let g = gcd(dx, dy). Then step is (dx/g, dy/g).
                                g = abs(dx)
                                h = abs(dy)
                                while g != 0 and h != 0:
                                    g, h = h, g % h
                                g = g if g != 0 else 1
                                sx = dx // g
                                sy = dy // g
                                
                                # Find t range
                                # -100 <= p1[0] + t*sx <= 100
                                # -100 <= p1[1] + t*sy <= 100
                                t_min = -1000
                                t_max = 1000
                                for t in range(t_min, t_max+1):
                                    x = p1[0] + t * sx
                                    y = p1[1] + t * sy
                                    if -100 <= x <= 100 and -100 <= y <= 100:
                                        p = [x, y]
                                        if p != old_p:
                                            candidates.add(tuple(p))
                    
                    # Add some random candidates
                    for _ in range(10):
                        x = random.randint(-100, 100)
                        y = random.randint(-100, 100)
                        candidates.add((x, y))
                    
                    for cx, cy in candidates:
                        new_p = [cx, cy]
                        if new_p in improved:
                            continue
                        improved[i] = new_p
                        if has_4_collinear(improved):
                            improved[i] = old_p
                            continue
                        new_score = count_3_collinear(improved)
                        if new_score > best_score:
                            best_score = new_score
                            best_new_p = new_p
                    improved[i] = old_p
                    
                    if best_new_p and best_score > current_score:
                        improved[i] = best_new_p
                        improved_flag = True
            
            return improved

        # Run a few local searches
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            init_pts = local_search_init()
            # Ensure no 4-collinear in init
            while has_4_collinear(init_pts):
                # Perturb
                i = random.randint(0, 19)
                init_pts[i] = [random.randint(-100, 100), random.randint(-100, 100)]
            
            improved_pts = improve(init_pts)
            if not has_4_collinear(improved_pts):
                candidates.append(improved_pts)
        
        # Also add some deterministic structured candidates
        # E.g., points on y = x^2 but with x in a specific set? No, no 3-collinear.
        # What about y = x^3? No.
        
        # Let's try a "staircase" pattern.
        stair = []
        for i in range(20):
            stair.append([i, i*i % 100])
        # Check 4-collinear
        if not has_4_collinear(stair):
            candidates.append(stair)
        
        return candidates

    best_config = None
    best_score = 21
    
    candidates = generate_candidates()
    
    for pts in candidates:
        if time.time() > deadline - 5:
            break
        if len(set(map(tuple, pts))) != 20:
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
        if has_4_collinear(pts):
            continue
        
        score = max_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
    if best_config is None:
        # Fallback: Random valid set
        while True:
            pts = []
            for _ in range(20):
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if [x, y] not in pts:
                        pts.append([x, y])
                        break
            if not has_4_collinear(pts):
                best_config = pts
                break
                
    return best_config

import random
