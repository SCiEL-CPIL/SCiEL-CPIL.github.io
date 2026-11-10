import random
import time
import math
from collections import defaultdict

def construct():
    N = 20
    BOUND = 100
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        return (A, B, C)

    def init_points():
        # Start with a grid-like pattern to have some structure
        # Or random
        pts = []
        # Try to place points on a few lines initially
        # For now, random within bounds
        while len(pts) < N:
            x = random.randint(-BOUND, BOUND)
            y = random.randint(-BOUND, BOUND)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts

    def count_triples(points):
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        triples = 0
        for c in line_counts.values():
            if c >= 2:
                triples += c * (c - 1) // 2
        return triples

    def is_valid_move(points, new_point):
        # Check if new_point can be added to points without creating 4 collinear
        # points is the set WITHOUT the point being moved (or before removal)
        # Actually, let's pass the set of OTHER points
        # For each other point q, check line(new_point, q)
        # Count how many points in points are on that line
        # If count >= 2, then adding new_point makes 4 (new + q + 2 others)
        # Wait. If line has q and 2 others, total 3 existing. Adding new makes 4.
        # So if count of existing points on line(new_point, q) is >= 2, invalid.
        # Note: q is one of the existing points.
        # So we check for each q in points:
        #   line_key = get_line_key(new_point, q)
        #   count = sum(1 for p in points if p != q and get_line_key(p, q) == line_key)
        #   This is O(N^2). Too slow for SA inner loop.
        
        # Optimized:
        # Precompute line counts for the current set 'points'?
        # The caller should maintain line_counts.
        pass

    # We will maintain line_counts externally for efficiency.
    
    def simulate_annealing():
        points = init_points()
        
        # Initialize line_counts
        line_counts = defaultdict(int)
        for i in range(N):
            for j in range(i + 1, N):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        
        current_triples = count_triples(points)
        best_points = points[:]
        best_triples = current_triples
        
        start_time = time.time()
        time_limit = 100 # Leave buffer
        
        T = 10.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick random index
            i = random.randint(0, N - 1)
            old_point = points[i]
            
            # Remove old_point from line_counts
            # Get all lines through old_point
            # We need to identify which lines change.
            # Iterate through all other points
            affected_lines = []
            for j in range(N):
                if i == j: continue
                key = get_line_key(old_point, points[j])
                line_counts[key] -= 1
                affected_lines.append(key)
            
            # Try to find a new point
            new_point = None
            attempts = 0
            max_attempts = 50
            valid_gain = -1
            
            while attempts < max_attempts:
                attempts += 1
                nx = random.randint(-BOUND, BOUND)
                ny = random.randint(-BOUND, BOUND)
                cand = (nx, ny)
                if cand in points:
                    continue
                
                # Check validity and gain
                # For each other point q, check line(cand, q)
                # We need to know how many points in points (excluding i, so effectively the current set minus old_point) are on that line.
                # Since we already removed old_point from line_counts, line_counts reflects the set of N-1 points.
                # So for line L through cand, the count of points in the current set (N-1) is line_counts[L].
                # If we add cand, the new count is line_counts[L] + 1.
                # We need new_count < 4 => line_counts[L] < 3.
                # So if any line through cand has line_counts[L] >= 3, it's invalid.
                
                valid = True
                gain = 0
                # To avoid O(N) check for every candidate, we can just iterate. N=19.
                for j in range(N):
                    if i == j: continue
                    key = get_line_key(cand, points[j])
                    # Check if this line has >= 3 points in the current set
                    if line_counts[key] >= 3:
                        valid = False
                        break
                    # Calculate gain contribution?
                    # Gain is sum of C(k, 2) for lines through cand.
                    # But we can just calculate total triples difference later.
                    # Or calculate gain now:
                    # For each unique line through cand, k = line_counts[line].
                    # New triples formed = C(k+1, 2) - C(k, 2) = k.
                    # Wait.
                    # Old triples on line L (with k points) = C(k, 2).
                    # New triples on line L (with k+1 points) = C(k+1, 2).
                    # Difference = k.
                    # So gain = sum of k for all unique lines through cand.
                    # We need to sum unique lines.
                
                if not valid:
                    continue
                
                # Calculate gain
                # We need unique lines through cand
                seen_lines = set()
                gain = 0
                for j in range(N):
                    if i == j: continue
                    key = get_line_key(cand, points[j])
                    if key not in seen_lines:
                        seen_lines.add(key)
                        gain += line_counts[key]
                
                if gain > valid_gain:
                    valid_gain = gain
                    new_point = cand
            
            if new_point is None:
                # Restore old point
                for key in affected_lines:
                    line_counts[key] += 1
                continue
            
            # Calculate delta
            delta = valid_gain - 0 # We didn't subtract old gain?
            # Wait. The "gain" calculated above is the number of NEW triples formed by adding new_point to the set of N-1 points.
            # The old triples involving old_point are already removed from line_counts?
            # No, line_counts was updated by removing old_point.
            # So current_triples should be updated?
            # It's easier to just recalculate total triples if accepted?
            # Or maintain current_triples.
            
            # Let's recalculate current_triples for the set of N-1 points? No, that's expensive.
            # Let's just use the gain as the delta relative to the state BEFORE removing old_point?
            # No.
            # State S: N points. Triples T.
            # Remove old_point: Triples T' = T - (triples involving old_point).
            # Add new_point: Triples T'' = T' + (triples involving new_point in S\{old}).
            # We calculated gain = triples involving new_point in S\{old}.
            # We need to know triples involving old_point in S.
            # Let's calculate old_gain = sum of k for lines through old_point in S\{old}.
            # This is symmetric.
            
            # To keep it simple and robust, let's just calculate the total triples of the new set if we accept?
            # N=20, O(N^2) = 400 ops. Very fast.
            # We can just try the move, calculate new_total, and compare.
            
            # Restore old point to points list temporarily? No, we have points list.
            # points[i] is still old_point in the list, but line_counts reflects removal.
            # Let's just create a new list for trial.
            trial_points = points[:]
            trial_points[i] = new_point
            
            # Check validity of trial_points (just in case)
            # And calculate new_triples
            new_triples = count_triples(trial_points)
            
            # Acceptance
            if new_triples > current_triples:
                # Accept
                points = trial_points
                current_triples = new_triples
                # Update line_counts
                # We already removed old_point. Now we need to add new_point.
                for j in range(N):
                    if i == j: continue
                    key = get_line_key(new_point, points[j])
                    line_counts[key] += 1
                # Restore old point lines? No, we removed them. We added new ones.
                # But we need to clear the "affected_lines" logic?
                # Actually, the loop above "Remove old_point" modified line_counts.
                # The loop above "Add new_point" (if accepted) modifies it further.
                # This is correct.
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = points[:]
            else:
                # Reject
                # Restore old_point to line_counts
                for key in affected_lines:
                    line_counts[key] += 1
                # points list is unchanged (still has old_point at i)
            
            # Cooling
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_points

    points = simulate_annealing()
    
    # Final verification and cleanup
    # Ensure distinct and bounds
    # Ensure no 4 collinear
    # If invalid, fallback to simple grid
    
    def is_valid(points):
        if len(points) != N: return False
        if len(set(points)) != N: return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100: return False
        # Check no 4 collinear
        for i in range(N):
            p = points[i]
            line_counts = defaultdict(int)
            for j in range(N):
                if i == j: continue
                key = get_line_key(p, points[j])
                line_counts[key] += 1
            for c in line_counts.values():
                if c >= 3:
                    return False
        return True

    if not is_valid(points):
        # Fallback
        points = []
        for i in range(N):
            points.append([i, i*i])
        # This parabola has no 3 collinear, so valid.
        
    return [[p[0], p[1]] for p in points]
