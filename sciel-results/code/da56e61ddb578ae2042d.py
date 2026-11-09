import random
import time
import sys
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin against 120s limit

    # Helper to normalize direction vector (dx, dy)
    def norm_dir(dx, dy):
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        return (dx, dy)

    def get_line_key(p1, p2):
        """Returns a canonical key for the line passing through p1 and p2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        a, b = norm_dir(dx, dy)
        # Line equation: a*x + b*y = c
        c = a * x1 + b * y1
        return (a, b, c)

    def has_four_collinear(points):
        """Check if any line contains 4 or more points."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
                # Optimization: if we see the same line key, we don't need to increment 
                # if we just want to know if count >= 4. 
                # Actually, pairs on the same line share the key.
                # Number of points k on a line generates C(k,2) pairs.
                # So if C(k,2) >= C(4,2)=6, then k>=4.
                # But simpler: just count pairs. If pairs >= 6, k could be 4.
                # Wait, if k=3, pairs=3. If k=4, pairs=6.
                # So if any key has count >= 6, we have >= 4 points?
                # Not exactly. If we have 4 points, we have 6 pairs.
                # If we have 3 points, we have 3 pairs.
                # So checking if any line has >= 6 pairs is a good heuristic for 4 collinear.
                # However, it's safer to just track the set of points per line or reconstruct.
                # Given n=20, O(N^2) is small. We can just count pairs and check max.
                if line_counts[key] >= 6:
                    return True
        # Wait, the above logic is slightly flawed for detection speed if we stop early.
        # Let's do a proper check.
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        for pts in line_points.values():
            if len(pts) >= 4:
                return True
        return False

    def count_triples(points):
        """Count the number of distinct collinear triples."""
        n = len(points)
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        triples = 0
        for pts in line_points.values():
            k = len(pts)
            if k >= 3:
                triples += k * (k - 1) * (k - 2) // 6
        return triples

    def max_no_three_collinear(points):
        """
        Exact Branch and Bound for the largest subset with no three collinear.
        Since N=20 is small, we can use a recursive approach with pruning.
        """
        n = len(points)
        # Precompute for each point i, the set of points j that are collinear with i and some k?
        # Actually, the constraint is "no three collinear".
        # This is equivalent to finding the maximum independent set in a 3-uniform hypergraph?
        # No, it's simpler. A subset is valid if it contains no 3 collinear points.
        # This is the "no-three-in-line" problem on a fixed set.
        
        # We can use a simple recursive backtracking with bitmasks.
        # State: set of available points, current size.
        # But N=20 is too big for naive 2^20 if we aren't careful.
        # However, the structure is sparse.
        
        # A better approach: 
        # The largest subset with no 3 collinear is the complement of the minimum vertex cover of the 3-uniform hypergraph of collinear triples?
        # No, that's not quite right.
        
        # Let's use a heuristic B&B.
        # Order points by degree (number of triples they participate in).
        # Greedy choice: Pick a point, remove all points that would form a triple with it and any other picked point?
        
        # Actually, for N=20, a simple randomized greedy with backtracking might work.
        # Or, we can rely on the fact that the search space is constrained by the geometry.
        
        # Let's implement a recursive function:
        # f(remaining_points, current_set)
        # But we want to MINIMIZE the score. The score IS the size of this largest subset.
        # So we need to compute this value for the final candidate to know its score?
        # The prompt says: "The score is the largest subset with no three collinear (exact branch and bound). Minimize."
        # This implies the *evaluator* does the B&B. My program just returns the points.
        # So I don't need to compute the exact score myself, only optimize the configuration.
        # However, using the triple count as a proxy is good, but maximizing triples doesn't guarantee minimizing the max no-3-collinear subset.
        # Example: If I have one line of 10 points (invalid, 4 collinear), but if I had lines of 3, it's valid.
        # If I have many disjoint triples, the largest no-3-collinear subset will likely be large (picking 1 from each triple).
        # To MINIMIZE the largest no-3-collinear subset, we want to MAXIMIZE the interference.
        # Ideally, we want a structure where every point is in many triples, forcing any large subset to contain a triple.
        
        # The objective to optimize is actually: min (max_{S subset P, no 3 collinear} |S|).
        # This is a hard problem.
        # However, "maximizing the number of collinear triples" is a strong heuristic.
        # Even stronger: "minimizing the independence number of the 3-collinear hypergraph".
        
        # Since I can't easily compute the exact B&B score inside the loop (it might be slow),
        # I will stick to maximizing the number of collinear triples, and perhaps also penalizing long lines (though 4 is forbidden).
        
        pass

    # Initial random valid configuration
    best_points = None
    best_triples = -1
    
    def random_valid_points():
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                return pts

    # Generate initial points
    # Try a few random starts
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        pts = random_valid_points()
        t = count_triples(pts)
        if best_points is None or t > best_triples:
            best_points = pts
            best_triples = t

    if best_points is None:
        return [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,0], [3,1], [3,2], [3,3], [0,3], [1,3], [2,3], [4,0], [4,1], [4,2], [4,3]] # Fallback (might be invalid)

    current_points = best_points
    current_triples = best_triples

    # Local Search
    # Strategy: Move a point to a location that creates a new triple with two others,
    # while maintaining validity (no 4 collinear).
    
    # Pre-compute lines defined by pairs of points to find "candidate" positions.
    # For a point P to be moved to Q, Q should ideally be on a line with two other points.
    
    while time.time() < deadline - 5:
        improved = False
        # Try moving each point
        for idx in range(20):
            if time.time() > deadline - 2:
                break
                
            old_point = current_points[idx]
            
            # Generate candidates
            candidates = set()
            
            # 1. Points that are on lines with 2 other points
            # For any pair (j, k) where j,k != idx, let L be the line.
            # Find integer points on L that are not currently occupied (except by j,k).
            # If we place idx at one such point, it forms a triple (idx, j, k).
            # We need to check if it creates 4 collinear.
            
            for j in range(20):
                if j == idx: continue
                for k in range(j + 1, 20):
                    if k == idx: continue
                    
                    p_j = current_points[j]
                    p_k = current_points[k]
                    
                    # Get line parameters
                    x1, y1 = p_j
                    x2, y2 = p_k
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0: continue
                    step_x = dx // g
                    step_y = dy // g
                    
                    # The line is infinite. We need to find integer points on this line
                    # that are within [-100, 100] and not already occupied by other points.
                    # We can iterate along the line.
                    # Range of t such that p_j + t * (step_x, step_y) is in bounds.
                    
                    # Start from a point far enough back
                    min_t = -200
                    max_t = 200
                    
                    for t in range(min_t, max_t + 1):
                        if t == 0 or t == 1:
                            continue # These are j and k
                        nx = x1 + t * step_x
                        ny = y1 + t * step_y
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            candidates.add((nx, ny))

            # 2. Random nearby points
            for _ in range(50):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                candidates.add((nx, ny))
                
            # Filter candidates: must be distinct from all other points
            existing_points = set(current_points)
            existing_points.remove(old_point)
            
            valid_candidates = []
            for cand in candidates:
                if cand not in existing_points:
                    valid_candidates.append(cand)
            
            random.shuffle(valid_candidates)
            
            # Try candidates
            best_cand = None
            best_cand_triples = current_triples
            
            for cand in valid_candidates:
                if time.time() > deadline - 2:
                    break
                new_points = current_points.copy()
                new_points[idx] = cand
                
                if has_four_collinear(new_points):
                    continue
                
                new_triples = count_triples(new_points)
                
                if new_triples > best_cand_triples:
                    best_cand_triples = new_triples
                    best_cand = cand
                    if new_triples > current_triples + 5: # Early exit if huge improvement
                        break
            
            if best_cand is not None:
                current_points[idx] = best_cand
                current_triples = best_cand_triples
                improved = True
                if current_triples > best_triples:
                    best_points = current_points.copy()
                    best_triples = current_triples
                
        if not improved:
            # Try a random restart or perturb if stuck?
            # Simple: if no improvement in a while, do a random restart from best
            # For now, just break if no improvement in this pass? No, try again with different random candidates.
            # To avoid infinite loop on same state, we can accept equal with small prob.
            if random.random() < 0.1:
                # Random perturbation
                idx = random.randint(0, 19)
                while True:
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    if (nx, ny) not in current_points:
                        break
                # Check validity
                new_points = current_points.copy()
                new_points[idx] = (nx, ny)
                if not has_four_collinear(new_points):
                    new_triples = count_triples(new_points)
                    # Accept if not worse
                    if new_triples >= current_triples:
                        current_points = new_points
                        current_triples = new_triples
                        if current_triples > best_triples:
                            best_points = current_points.copy()
                            best_triples = current_triples

    return [[p[0], p[1]] for p in best_points]

# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Count:", len(pts))
    print("Has 4 collinear:", has_four_collinear(pts))
    print("Triples:", count_triples(pts))
