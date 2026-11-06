import time
import random
import sys

# EVOLVE-BLOCK-START

def cross(p1, p2, p3):
    """
    Returns the cross product of vectors (p2-p1) and (p3-p1).
    If 0, points are collinear.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def check_no_four_collinear(points):
    """
    Checks if any 4 points are collinear.
    Returns True if valid (no 4 collinear).
    """
    n = len(points)
    # Map line defined by pair (i, j) to count of points on it
    # To be efficient, we only need to check lines defined by pairs.
    # If a line has >= 4 points, it's invalid.
    # We can just count points on lines defined by pairs (0,1), (0,2)... 
    # Actually, simpler: for each pair (i, j), count how many k are collinear.
    # If any count >= 4, return False.
    # This is O(N^3) which is 20^3 = 8000 ops, very fast.
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if is_collinear(points[i], points[j], points[k]):
                    count += 1
            if count >= 4:
                return False
    return True

def count_collinear_triples(points):
    """
    Counts the number of collinear triples in the set.
    This serves as a heuristic proxy for 'density'.
    Higher density usually implies a smaller max no-3-collinear subset.
    """
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def estimate_score(points):
    """
    Heuristic estimate of the largest subset with no 3 collinear.
    We use a greedy approach to find a large such subset.
    This is an upper bound on the true score.
    """
    n = len(points)
    # Greedy: pick a point, remove all points that form a collinear triple with it and another picked point?
    # Actually, simpler greedy for Max No-3-Collinear:
    # 1. Start with empty set S.
    # 2. Iterate through points.
    # 3. If adding point p to S keeps S no-3-collinear, add it.
    # This finds a valid subset, but not necessarily the maximum.
    # To get a better estimate, we can try multiple random orderings.
    
    best_size = 0
    # Try a few random permutations to approximate the maximum
    # For speed, we'll just do one greedy pass with a specific order (e.g., index)
    # and maybe one random.
    
    def greedy_max_no3(order):
        S = []
        for idx in order:
            p = points[idx]
            valid = True
            # Check if p forms a collinear triple with any two points in S
            # If S has size < 2, it's always valid
            if len(S) >= 2:
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if is_collinear(p, S[i], S[j]):
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                S.append(p)
        return len(S)
        
    # Standard order
    best_size = max(best_size, greedy_max_no3(list(range(n))))
    
    # Random order (1-2 times for speed)
    indices = list(range(n))
    random.shuffle(indices)
    best_size = max(best_size, greedy_max_no3(indices))
    
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Precompute all grid points
    grid_points = [[x, y] for x in range(-100, 101) for y in range(-100, 101)]
    
    best_score = 20
    best_points = None
    
    # Greedy construction with restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        current_points = []
        
        # Randomly select a starting point to break symmetry
        if num_restarts > 1:
            start_p = random.choice(grid_points)
        else:
            start_p = [0, 0]
            
        current_points.append(start_p)
        
        # Keep track of lines defined by pairs in current_points to enforce no-4-collinear
        # line_key: (slope_num, slope_den, intercept_num, intercept_den) -> count
        # But integer lines are tricky. 
        # Alternative: For each candidate, check against all pairs in current_points.
        # Since N=20, pairs <= 190. Candidates <= 40000. 190*40000 = 7.6M ops per step.
        # 20 steps * 7.6M = 150M ops. This might be too slow in Python.
        
        # Optimization: 
        # We only need to check if the candidate creates a 4-collinear.
        # A 4-collinear means candidate lies on a line defined by 2 existing points 
        # AND that line already contains 2 other existing points? No.
        # If candidate lies on line L defined by p1, p2.
        # If L contains k other points from current_points, total is k+2.
        # We need total < 4, so k < 2.
        # So we need to know, for each line defined by a pair in current_points, 
        # how many points from current_points lie on it.
        
        # Let's maintain a dictionary: line_rep -> set of point indices on that line
        # Line representation: (A, B, C) for Ax + By + C = 0, normalized.
        
        def get_line_rep(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = -(A * p1[0] + B * p1[1])
            # Normalize by GCD
            import math
            g = math.gcd(A, math.gcd(B, C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            return (A, B, C)
        
        lines = {} # rep -> list of indices
        
        # Add start point
        lines = {} # No pairs yet
        
        for step in range(1, 20):
            best_candidate = None
            best_triples = -1
            best_random = random.random()
            
            # Sample candidates to speed up? No, 40k is manageable if optimized.
            # But 20 steps * 40k candidates * 190 pairs check = 150M checks.
            # Each check is a cross product.
            # We can prune: 
            # For a candidate p, it is invalid if it lies on a line that already has 2 points.
            # We can pre-compute for each candidate? No, lines change.
            
            # Faster check:
            # Iterate through existing lines.
            # If a line has count >= 2, candidate must NOT be on it.
            # If a line has count == 1, candidate CAN be on it (becomes 2).
            # If a line has count == 0, candidate CAN be on it (becomes 1, but wait, lines are defined by pairs).
            
            # Actually, simpler:
            # For candidate p:
            # 1. Check if p is on any line defined by 2 points in current_points that already has 2 points.
            #    If yes, invalid.
            # 2. Count how many new collinear triples it forms.
            
            # Let's iterate candidates.
            # To save time, we can skip points that are "obviously" bad?
            
            # Optimization: Only consider points that are not collinear with any pair of existing points 
            # that form a line with 2+ points.
            
            # Let's try a subset of candidates to keep it fast, or just optimize the inner loop.
            
            # We'll iterate all grid points.
            for p in grid_points:
                if p in current_points:
                    continue
                
                # Check validity (no 4 collinear)
                valid = True
                # Check against all lines that currently have >= 2 points
                # We need to know which lines have >= 2 points.
                # Let's maintain a list of "full" lines (count >= 2).
                
                # Actually, let's just check against all pairs (i, j) in current_points.
                # If is_collinear(points[i], points[j], p) is True:
                #    Check if there are 2 other points on this line.
                #    This is expensive.
                
                # Better: Maintain `line_counts`.
                # For each pair (i, j), we have a line rep.
                # We can pre-compute for the *current* set of points, which lines have count 2.
                # Let's do this once per step.
                
                # Pre-compute "blocked" lines for this step
                blocked_lines = set()
                # And "potential" lines for triple counting
                # Actually, for triple counting, we just need to check collinearity with pairs.
                
                # Let's refine the loop.
                # 1. Pre-compute set of line reps that have >= 2 points in current_points.
                #    If candidate lies on such a line, it's invalid.
                
                # Re-calculate blocked lines every step? 
                # There are ~190 lines max. 
                # For each line, check if count >= 2.
                
                # Let's just do the O(N^3) check for validity on the final set, 
                # and during construction, use a simpler heuristic.
                
                # Heuristic: 
                # 1. Check if p is collinear with any 2 points that are already collinear with a 3rd?
                #    No, just check if p + any 2 existing points form a line with a 4th existing point.
                
                # Let's stick to the O(N^2) check per candidate against existing pairs, 
                # but break early.
                
                # For candidate p:
                # for i in range(len(current_points)):
                #    for j in range(i+1, len(current_points)):
                #       if is_collinear(pts[i], pts[j], p):
                #          # Check if there is a k != i, j such that is_collinear(pts[i], pts[j], pts[k])
                #          # This is O(N).
                #          # Total O(N^3) per candidate. Too slow.
                
                # Alternative:
                # Maintain `line_to_points` dict.
                # When adding a point, update it.
                # But we are evaluating candidates, not adding yet.
                
                # Let's pre-compute for the current set:
                # `dense_lines`: set of line reps where count >= 2.
                # If candidate is on a dense line, invalid.
                
                dense_lines = set()
                line_counts = {}
                pts = current_points
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        rep = get_line_rep(pts[i], pts[j])
                        if rep not in line_counts:
                            line_counts[rep] = 0
                        line_counts[rep] += 1
                        # Note: A line with k points has k*(k-1)/2 pairs.
                        # So if count of pairs is C, k is approx sqrt(2C).
                        # If k>=4, pairs >= 6.
                        # If k==3, pairs == 3.
                        # If k==2, pairs == 1.
                        # So if line_counts[rep] >= 3, then k>=3.
                        # If line_counts[rep] >= 6, then k>=4 (invalid).
                        # Wait, we want to prevent k=4.
                        # So if adding p makes k=4, it means p is on a line with k=3 existing points.
                        # So if line_counts[rep] == 3, the line has 3 points. Adding p makes 4. Invalid.
                        # If line_counts[rep] >= 6, the line already has >=4 points. But we check validity at end.
                        # During construction, we assume current set is valid (no 4).
                        # So max k is 3.
                        # So if line_counts[rep] == 3, the line has 3 points. Candidate cannot be on it.
                        
                        if line_counts[rep] == 3:
                            dense_lines.add(rep)
                
                # Now check candidate
                # If p is on any dense line, invalid.
                # Also, we need to count triples.
                
                # To speed up, we can skip candidates that are on dense lines.
                # How to check if p is on a line rep (A,B,C)? A*p[0] + B*p[1] + C == 0.
                
                # Let's iterate candidates.
                # We'll sample 1000 random candidates to keep it fast, or all if fast enough.
                # 40k candidates * (check dense lines + count triples)
                # Check dense lines: ~100 lines. 100 ops.
                # Count triples: Check collinearity with all pairs? No.
                # Just check if p forms a triple with any pair.
                # That's O(N^2) = 400 ops.
                # Total ~500 ops per candidate.
                # 40k * 500 = 20M ops per step.
                # 20 steps * 20M = 400M ops.
                # Python does ~10-50M simple ops/sec.
                # This will take 10-40 seconds per restart.
                # Too slow for multiple restarts in 110s.
                
                # Optimization:
                # We don't need to check ALL candidates.
                # We can prioritize candidates that are "close" to existing points or form lines.
                # Or, we can reduce the grid size? No, must be <= 100.
                
                # Let's try a smaller candidate set: 
                # Only consider points that are collinear with at least one existing point?
                # No, that biases towards lines.
                
                # Let's try random sampling of 500 candidates per step.
                # This reduces cost by 80x.
                
                candidates = random.sample(grid_points, 500)
                best_candidate = None
                best_triples = -1
                
                for p in candidates:
                    if p in current_points:
                        continue
                    
                    # Check dense lines
                    invalid = False
                    for rep in dense_lines:
                        A, B, C = rep
                        if A * p[0] + B * p[1] + C == 0:
                            invalid = True
                            break
                    if invalid:
                        continue
                    
                    # Count triples
                    triples = 0
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            if is_collinear(pts[i], pts[j], p):
                                triples += 1
                            
                    if triples > best_triples:
                        best_triples = triples
                        best_candidate = p
                        
                if best_candidate is None:
                    # Fallback: pick any valid point
                    for p in grid_points:
                        if p in current_points:
                            continue
                        invalid = False
                        for rep in dense_lines:
                            A, B, C = rep
                            if A * p[0] + B * p[1] + C == 0:
                                invalid = True
                                break
                        if not invalid:
                            best_candidate = p
                            break
                
                if best_candidate is None:
                    # Restart
                    break
                
                current_points.append(best_candidate)
                
                # Update lines dict for next step? 
                # We recalculate dense_lines from scratch each step.
                # This is O(N^2) which is fine.
                
        if len(current_points) == 20:
            if check_no_four_collinear(current_points):
                score = estimate_score(current_points)
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in current_points]
                    # Try to improve with local search?
                    # For now, just record.
                    
        # If we ran out of time, break
        if time.time() > deadline - 5:
            break

    if best_points is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points

# EVOLVE-BLOCK-END
