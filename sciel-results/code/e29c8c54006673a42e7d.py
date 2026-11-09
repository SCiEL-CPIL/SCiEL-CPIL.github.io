import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety
    
    def collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def has_four_collinear(points):
        n = len(points)
        # Check all pairs, count points on the line
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = points[i], points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def max_no_three_collinear(points):
        """
        Exact Branch and Bound for the largest subset with no three collinear.
        Returns the size of the largest such subset.
        """
        n = len(points)
        if n <= 2:
            return n
            
        # Precompute collinearity masks
        # coll_mask[i][j] is a bitmask of points k such that i, j, k are collinear
        # Actually, for the check "can I add point idx to current_set?",
        # I need to know if there exist a, b in current_set such that idx, a, b are collinear.
        # This is equivalent to: for each a in current_set, is there a b in current_set 
        # such that b is in collinear_with_pair(a, idx)?
        
        # Let's precompute pair_collinear[i][j] = bitmask of k (k != i, k != j) collinear with i, j.
        # If we add idx, we need to check if for any a in current_set, 
        # (pair_collinear[a][idx] & current_mask) is not empty.
        
        pair_collinear = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
                pair_collinear[i][j] = mask
        
        best_size = [0]
        
        # Order points by "collinearity density" to prune earlier?
        # Points with many collinear partners are "harder" to include, 
        # but including them early might block others.
        # Standard order 0..n-1 is fine, but sorting by degree might help.
        # Let's stick to 0..n-1 for simplicity and determinism in B&B, 
        # but the bitmask check is fast.
        
        def backtrack(idx, current_mask, current_size):
            # Pruning
            if current_size + (n - idx) <= best_size[0]:
                return
            
            if idx == n:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                return
            
            # Option 1: Exclude point idx
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include point idx
            # Check if idx can be added
            # We need to ensure no a, b in current_mask are collinear with idx.
            # i.e., for all a in current_mask, pair_collinear[a][idx] & current_mask == 0
            can_add = True
            temp_mask = current_mask
            while temp_mask:
                a_bit = temp_mask & (-temp_mask)
                a = a_bit.bit_length() - 1
                # If there is any b in current_mask that is collinear with a and idx
                if pair_collinear[a][idx] & current_mask:
                    can_add = False
                    break
                temp_mask ^= a_bit
            
            if can_add:
                backtrack(idx + 1, current_mask | (1 << idx), current_size + 1)
        
        backtrack(0, 0, 0)
        return best_size[0]

    # Search Strategy
    best_score = 20
    best_points = None
    
    # 1. Initial Heuristic Construction: "Line Packing"
    # Try to create 5 lines of 4 points each.
    # Lines: y=0, y=1, y=2, x=0, x=1
    # Points:
    # L1 (y=0): (-10, 0), (-5, 0), (5, 0), (10, 0)
    # L2 (y=1): (-10, 1), (-5, 1), (5, 1), (10, 1)
    # L3 (y=2): (-10, 2), (-5, 2), (5, 2), (10, 2)
    # L4 (x=0): (0, -10), (0, -5), (0, 5), (0, 10)
    # L5 (x=1): (1, -10), (1, -5), (1, 5), (1, 10)
    
    # Check for 4-collinear in this construction.
    # Intersections:
    # (0,0) is on L1 and L4. But we don't have (0,0) in the list.
    # L1 has y=0. L4 has x=0.
    # Points on L1: x in {-10, -5, 5, 10}. None are 0.
    # Points on L4: y in {-10, -5, 5, 10}. None are 0.
    # So no intersection points are shared in the list?
    # Wait, L1 is y=0. L4 is x=0.
    # Are there any points that are on both? No, because x!=0 for L1 points and y!=0 for L4 points.
    # What about L1 and L2? Parallel.
    # What about L4 and L5? Parallel.
    # What about L1 and L5? L1 is y=0, L5 is x=1. Intersection (1,0).
    # Is (1,0) in L1? No (x=1 not in {-10,-5,5,10}).
    # Is (1,0) in L5? No (y=0 not in {-10,-5,5,10}).
    # So this construction has NO intersections among the selected points?
    # Let's verify 4-collinear.
    # Any line containing 3 points from this set?
    # The points are chosen such that any line through 2 points contains no other points from the set,
    # EXCEPT the lines we defined.
    # On L1, we have 4 points. That's a 4-collinear set!
    # Constraint: "no four collinear".
    # So this initial guess is INVALID.
    
    # We need to break the 4-collinear sets.
    # Strategy: Take these 20 points, and perturb one point from each 4-set to a "random" location
    # such that it doesn't create new 4-collinear sets and ideally creates new 3-collinear sets
    # with points from other lines.
    
    initial_points = [
        (-10, 0), (-5, 0), (5, 0), (10, 0),
        (-10, 1), (-5, 1), (5, 1), (10, 1),
        (-10, 2), (-5, 2), (5, 2), (10, 2),
        (0, -10), (0, -5), (0, 5), (0, 10),
        (1, -10), (1, -5), (1, 5), (1, 10)
    ]
    
    # Perturb one point from each group of 4 to break the 4-collinearity.
    # Groups: 0-3, 4-7, 8-11, 12-15, 16-19
    # Move index 3, 7, 11, 15, 19.
    
    current_points = [list(p) for p in initial_points]
    
    # Define target positions for the moved points to create 3-collinearities with other lines
    # e.g. Move (10,0) to (2, 3)?
    # Let's just randomize the moved points within a small range and check validity.
    
    def try_local_search(points, iterations=2000):
        points = [list(p) for p in points]
        if has_four_collinear(points):
            return None, 20
        
        score = max_no_three_collinear(points)
        best_local_score = score
        best_local_points = [list(p) for p in points]
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 19)
            old_p = points[idx][:]
            
            # Try a few random moves
            found_move = False
            for _ in range(5):
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p == old_p:
                    continue
                
                # Check validity: no 4 collinear
                # Efficient check:
                # 1. Is new_p collinear with any 3 existing points?
                valid = True
                for i, j, k in itertools.combinations(range(20), 3):
                    if i == idx or j == idx or k == idx:
                        continue
                    if collinear(tuple(new_p), tuple(points[i]), tuple(points[j])) and \
                       collinear(tuple(new_p), tuple(points[i]), tuple(points[k])):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Apply move
                points[idx] = new_p
                
                # Calculate score
                # Optimization: Only recalculate if we expect improvement?
                # For now, calculate exact score.
                new_score = max_no_three_collinear(points)
                
                if new_score < score:
                    score = new_score
                    if score < best_local_score:
                        best_local_score = score
                        best_local_points = [list(p) for p in points]
                    found_move = True
                    break
                else:
                    # Revert
                    points[idx] = old_p
            
            if not found_move:
                # If no improvement in this iteration, try a larger jump?
                # Or just continue.
                pass
        
        return best_local_points, best_local_score

    # Run local search from the perturbed initial configuration
    # We need to perturb the initial points to make them valid first.
    # Let's create a valid starting point by moving the 5 "extra" points.
    
    valid_start = None
    attempts = 0
    while valid_start is None and attempts < 100:
        attempts += 1
        cand = [list(p) for p in initial_points]
        # Move indices 3, 7, 11, 15, 19
        for idx in [3, 7, 11, 15, 19]:
            cand[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if not has_four_collinear(cand):
            valid_start = cand
            break
    
    if valid_start is None:
        # Fallback to random valid configuration
        while True:
            cand = []
            while len(cand) < 20:
                p = [random.randint(-100, 100), random.randint(-100, 100)]
                if p not in cand:
                    cand.append(p)
            if not has_four_collinear(cand):
                valid_start = cand
                break
    
    # Evaluate and optimize
    best_points, best_score = try_local_search(valid_start, iterations=1000)
    
    # Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Generate a new random valid configuration
        cand = []
        while len(cand) < 20:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if p not in cand:
                cand.append(p)
        
        # Quick fix for 4-collinear
        valid = False
        for _ in range(100):
            if not has_four_collinear(cand):
                valid = True
                break
            idx = random.randint(0, 19)
            cand[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if not valid:
            continue
            
        pts, score = try_local_search(cand, iterations=500)
        if score < best_score:
            best_score = score
            best_points = pts

    # Final safety check
    if best_points is None:
        # Return a known valid configuration (parabola)
        best_points = [[x, x*x] for x in range(-10, 10)]
    
    # Ensure return type is list of lists of ints
    return [[int(x), int(y)] for x, y in best_points]

# Run construct
print(construct())
