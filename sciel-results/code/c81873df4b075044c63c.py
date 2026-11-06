"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    # We need exactly 20 distinct integer points with |x|,|y| <= 100.
    # Constraint: no four collinear.
    # Objective: minimize the size of the largest subset with no three collinear.
    # Let S be the set of 20 points.
    # Let H be the hypergraph of collinear triples.
    # We want to maximize the number of triples covered by the points, effectively
    # making the "independent set" (no 3 collinear) as small as possible.
    
    # A point is "independent" from a set T if it doesn't form a triple with any pair in T.
    # Actually, the score is the maximum size of a subset I such that no 3 points in I are collinear.
    
    # Strategy:
    # 1. Generate a candidate set of 20 points.
    # 2. Verify no 4 are collinear.
    # 3. Calculate the score (max independent set size).
    # 4. Use local search to improve (minimize score).
    
    # To get a good initial guess, we can try to place points on a grid or random.
    # However, random points often have no collinear triples at all (score 20).
    # We need to *force* collinear triples to reduce the max independent set.
    
    # Let's start with a known good configuration or generate one via greedy.
    # Greedy: Pick points one by one.
    # This is hard to do directly.
    
    # Alternative: Start with a random set, then try to "break" independence by moving points.
    # Or better: Construct points to lie on many lines.
    
    # Let's try a construction based on a grid but with perturbations to avoid 4-collinear.
    # A 5x4 grid has 20 points.
    # Points: (x, y) where x in {0,1,2,3,4}, y in {0,1,2,3}.
    # Check 4-collinear:
    # Horizontal lines: 4 points each. -> Violation!
    # We need to break these.
    
    # Let's use a "twisted" grid or a specific known pattern.
    # Or use a random search with a strong local improvement.
    
    # Let's implement the scoring function first.
    
    def get_lines(points):
        """Return a list of lines (frozensets of indices) that contain >= 3 points."""
        n = len(points)
        lines = []
        # Use a set of normalized line coefficients (A, B, C) for Ax+By+C=0
        # To avoid floating point issues, use integer arithmetic.
        # Line through (x1,y1) and (x2,y2):
        # (y2-y1)x - (x2-x1)y + (x2-y2)x1 + (y2-x1)y1 = 0? No.
        # A = y2 - y1
        # B = x1 - x2
        # C = x2*y1 - x1*y2
        # Normalize by GCD(A, B, C) and sign.
        
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2*y1 - x1*y2
                # Normalize
                if A == 0 and B == 0:
                    continue # Same point, shouldn't happen
                g = _gcd3(A, B, C)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                # Ensure sign consistency: first non-zero among A, B is positive
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                line_map[key].add(i)
                line_map[key].add(j)
        
        for key, pts in line_map.items():
            if len(pts) >= 3:
                lines.append(frozenset(pts))
        return lines

    def _gcd3(a, b, c):
        import math
        g = math.gcd(a, b)
        g = math.gcd(g, c)
        if g == 0:
            return 1
        return abs(g)

    def is_valid(points):
        """Check if no 4 points are collinear."""
        lines = get_lines(points)
        for line in lines:
            if len(line) >= 4:
                return False
        return True

    def calc_score(points):
        """
        Calculate the largest subset with no three collinear.
        This is the size of the maximum independent set in the 3-uniform hypergraph
        where hyperedges are the collinear triples.
        """
        n = len(points)
        lines = get_lines(points)
        # Extract all triples from lines
        triples = set()
        for line in lines:
            if len(line) == 3:
                triples.add(frozenset(line))
            elif len(line) == 4:
                # If we had 4, it's invalid, but let's handle it just in case
                # A line with 4 points contains 4 choose 3 = 4 triples.
                pts = list(line)
                for i in range(4):
                    for j in range(i+1, 4):
                        for k in range(j+1, 4):
                            triples.add(frozenset([pts[i], pts[j], pts[k]]))
        
        if not triples:
            return n
        
        # Branch and Bound for Maximum Independent Set in Hypergraph
        # Actually, we want the largest set I such that NO triple is fully contained in I.
        # This is the complement of the minimum hitting set for triples? No.
        # It's the maximum subset avoiding all hyperedges.
        
        # Let's use a simple B&B.
        # State: current set of indices, next index to consider.
        # We can prune if current_size + (n - next_idx) <= best_found.
        
        best = 0
        # Sort triples by some heuristic? Maybe by frequency of points?
        # Let's just iterate.
        
        # Optimization: If a point is in many triples, excluding it might be better?
        # Actually, we want to INCLUDE points. We exclude a point only if it completes a triple.
        
        # Let's define the search:
        # Can we add point i to the current independent set S?
        # Yes, if for all triples T, T is not a subset of S U {i}.
        # This is equivalent to: for all pairs {j, k} in S, {j, k, i} is not a triple.
        
        # Precompute: for each pair (i, j), list of k such that {i, j, k} is a triple.
        pair_to_third = defaultdict(list)
        for t in triples:
            pts = list(t)
            pair_to_third[frozenset([pts[0], pts[1]])].append(pts[2])
            pair_to_third[frozenset([pts[0], pts[2]])].append(pts[1])
            pair_to_third[frozenset([pts[1], pts[2]])].append(pts[0])
        
        # Convert to sets for faster lookup
        pair_to_third_sets = {k: set(v) for k, v in pair_to_third.items()}
        
        def bnb(idx, current_set, current_size):
            nonlocal best
            if current_size > best:
                best = current_size
            
            # Pruning
            if current_size + (n - idx) <= best:
                return
            
            if idx == n:
                return
            
            # Option 1: Exclude point idx
            bnb(idx + 1, current_set, current_size)
            
            # Option 2: Include point idx
            # Check if adding idx creates a triple with any pair in current_set
            conflict = False
            for p in current_set:
                key = frozenset([p, idx])
                if key in pair_to_third_sets:
                    # If any of the third points is also in current_set, conflict
                    for q in pair_to_third_sets[key]:
                        if q in current_set:
                            conflict = True
                            break
                if conflict:
                    break
            
            if not conflict:
                current_set.add(idx)
                bnb(idx + 1, current_set, current_size + 1)
                current_set.remove(idx)
        
        bnb(0, set(), 0)
        return best

    # Initial Configuration
    # Let's try to construct a set with many collinear triples.
    # A known small cap set or similar?
    # For n=20, the max independent set size is likely around 10-12?
    # Random points give 20.
    
    # Let's start with a random set and try to improve.
    # But random sets have 0 triples. Score 20.
    # We need to introduce triples.
    
    # Heuristic: Place points on a few lines.
    # E.g., 4 lines of 5 points?
    # If we have 4 lines, each with 5 points, and they intersect at 1 point?
    # Total points = 4*5 - 3*1 (intersections) ... complicated.
    
    # Let's try a "grid with noise" approach.
    # Take a 5x4 grid, and perturb points slightly to break 4-collinearities while keeping 3-collinearities?
    # Integer points only.
    
    # Let's try a specific construction:
    # Points on y = x^2 are no 3 collinear.
    # Points on a line are all collinear.
    
    # Let's use a randomized local search.
    # 1. Start with a random valid set (no 4 collinear).
    # 2. Calculate score.
    # 3. Try to swap a point with another to reduce score.
    
    # Generating a random valid set is easy: just pick random points and check.
    # But most random sets have score 20.
    
    # To get a lower score, we need to *force* collinearities.
    # Let's start with a set of points that has some structure.
    # E.g., take 4 lines. Place 5 points on each line.
    # Ensure no 4 are collinear (so lines must have exactly 3 or 4 points? No, constraint is NO 4 collinear. So lines can have 3 points max).
    
    # So we want to pack as many 3-collinear triples as possible into 20 points.
    # Each point can be in multiple triples.
    
    # Let's try a constructive approach:
    # Start with 0 points.
    # Add points one by one.
    # When adding a point, try to place it such that it forms a triple with existing pairs.
    
    # This is a greedy algorithm.
    # 1. Pick first 2 points.
    # 2. Pick 3rd point to form a triple with 1 and 2? No, 1 and 2 define a line.
    #    If we place 3 on that line, we have a triple.
    #    But we can't have 4 on that line.
    # 3. So, pick lines, and place points on them.
    
    # Let's try to generate a set with many triples.
    # We'll use a "pool" of candidate lines.
    
    # Simpler: Use a known good configuration from literature or search.
    # Since I don't have one, I will implement a local search that starts from a "structured" random set.
    
    # Structured random:
    # Divide the plane into 4 quadrants.
    # Place points on a few diagonal lines.
    
    # Let's just do a robust random restart local search.
    # 1. Generate a random set of 20 points.
    # 2. If invalid (4 collinear), repair it by moving one point.
    # 3. Calculate score.
    # 4. Local search: Try to replace a point p with a new point q.
    #    - q must keep the set valid (no 4 collinear).
    #    - q should ideally create new triples or break independence.
    #    - Accept if score decreases.
    
    # To speed up, we can precompute lines for the current set.
    
    best_score = 20
    best_points = None
    
    # Helper to repair a set
    def repair(points):
        """Ensure no 4 collinear by moving points if necessary."""
        points = [list(p) for p in points]
        for _ in range(100):
            lines = get_lines(points)
            bad = [l for l in lines if len(l) >= 4]
            if not bad:
                return points
            # Pick a random bad line
            line = random.choice(bad)
            # Pick a random point on this line to move
            idx = random.choice(list(line))
            # Move it to a random new position
            # Try to find a position that doesn't create new 4-collinearities
            for _ in range(50):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if [nx, ny] in points:
                    continue
                points[idx] = [nx, ny]
                if is_valid(points):
                    return points
            # If failed, revert
            # This is a simplified repair, might fail.
            # If it fails, we just return the original (which might be invalid)
            # But we check validity before using.
        return points

    # Generate initial candidate
    def generate_candidate():
        # Try to create a set with some collinearities
        # Pick 5 random lines
        lines = []
        for _ in range(10):
            # Random line in range
            x1 = random.randint(-100, 100)
            y1 = random.randint(-100, 100)
            x2 = random.randint(-100, 100)
            y2 = random.randint(-100, 100)
            if x1 == x2 and y1 == y2:
                continue
            lines.append((x1, y1, x2, y2))
        
        points = []
        # Place points on these lines
        for line in lines:
            x1, y1, x2, y2 = line
            # Find integer points on this line
            # Parametric: x = x1 + t*dx, y = y1 + t*dy
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(dx, dy)
            if g == 0:
                continue
            step_x = dx // g
            step_y = dy // g
            
            # Find range of t
            # We just pick a few random t values
            candidates = []
            for t in range(-50, 51):
                x = x1 + t * step_x
                y = y1 + t * step_y
                if -100 <= x <= 100 and -100 <= y <= 100:
                    if [x, y] not in points:
                        candidates.append([x, y])
            
            if candidates:
                # Pick 2-3 points from this line
                k = min(3, len(candidates), 20 - len(points))
                if k > 0:
                    chosen = random.sample(candidates, k)
                    points.extend(chosen)
        
        # Fill remaining with random points
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in points:
                points.append([x, y])
        
        # Ensure exactly 20
        points = points[:20]
        
        # Repair
        points = repair(points)
        return points

    # Local Search
    def local_search(points):
        points = [list(p) for p in points]
        current_score = calc_score(points)
        best_local_score = current_score
        best_local_points = [list(p) for p in points]
        
        no_improve = 0
        while time.time() < deadline - 5.0 and no_improve < 20:
            # Pick a random point to replace
            idx = random.randint(0, 19)
            old_point = points[idx]
            
            # Try to find a new point
            improved = False
            for _ in range(50):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if [nx, ny] in points:
                    continue
                
                new_points = [list(p) for p in points]
                new_points[idx] = [nx, ny]
                
                if not is_valid(new_points):
                    continue
                
                new_score = calc_score(new_points)
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_points = new_points
                    points = new_points
                    current_score = new_score
                    no_improve = 0
                    improved = True
                    break
            if not improved:
                no_improve += 1
        
        return best_local_points, best_local_score

    # Main loop
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        # Generate candidate
        points = generate_candidate()
        if not is_valid(points):
            continue
            
        # Local search
        improved_points, score = local_search(points)
        
        if score < best_score:
            best_score = score
            best_points = improved_points
            print(f"Restart {restart}: Improved score to {best_score}")
            
        if best_score <= 10: # Good enough?
            break

    if best_points is None:
        # Fallback: Random valid set
        while True:
            points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
            # Ensure distinct
            if len(set(map(tuple, points))) == 20:
                if is_valid(points):
                    return points

    return best_points
# EVOLVE-BLOCK-END
