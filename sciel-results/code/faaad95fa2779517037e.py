import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k = 3
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    # Encode point (xa, xb, ya, yb) to index 0..2400
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    # Decode index to (xa, xb, ya, yb)
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line_id[point_idx, dir_idx]
    # dir_idx 0..48: m = (ma, mb), ma = dir_idx // p, mb = dir_idx % p
    # dir_idx 49: vertical (0, 1)
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    # Generate all points and their coordinates
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
    # Fill line_id
    # For direction m, line_id = y - m*x
    # We can compute this vectorized if we represent field elements as integers 0..48
    # Let's map (a, b) to int a*7+b
    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Precompute field multiplication table for speed? 
    # Actually, let's just compute line_id in a loop but vectorize the inner part if possible.
    # Since n_points=2401 and n_dirs=50, total ops ~ 120k, which is fast even in Python.
    
    for d in range(n_dirs):
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d] = field_to_int(c)
        else:
            # Vertical direction (0,1), line is x = const
            # line_id = x
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Validation function using vectorized numpy
    # S is a boolean array of size n_points
    # Condition: For every point y, exists d such that:
    #   if S[y]: count(S on line_id[y,d]) == 1
    #   if not S[y]: count(S on line_id[y,d]) == 0
    
    def is_valid_vectorized(S):
        satisfied = np.zeros(n_points, dtype=bool)
        
        for d in range(n_dirs):
            lids = line_id[:, d]
            # Count S points on each line
            # We need counts per line ID. Line IDs are 0..48.
            counts = np.bincount(lids[S], minlength=q)
            
            # For each point y, get the count of S on its line for direction d
            counts_for_points = counts[lids]
            
            # Check condition
            # If S[y] is True, we need counts_for_points[y] == 1
            # If S[y] is False, we need counts_for_points[y] == 0
            valid_for_d = (S & (counts_for_points == 1)) | (~S & (counts_for_points == 0))
            
            satisfied |= valid_for_d
            
            if np.all(satisfied):
                return True
                
        return False

    def count_satisfied(S):
        """Returns the number of points that are 'satisfied' by the current S."""
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            counts = np.bincount(lids[S], minlength=q)
            counts_for_points = counts[lids]
            valid_for_d = (S & (counts_for_points == 1)) | (~S & (counts_for_points == 0))
            satisfied |= valid_for_d
        return np.sum(satisfied)

    # Initial S: Random
    # Start with a small random S to allow growth, or empty.
    # Let's try starting with a random set of size ~100-200.
    
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    best_S = None
    best_score = -1
    
    # Strategy: Simulated Annealing / Random Local Search
    # We want to maximize |S| subject to validity.
    # Actually, the metric is 2401 - |N| = |S|. So we want to maximize |S|.
    # The constraint is that S must be a "Nikodym complement".
    
    # Let's use a greedy approach with vectorized checks for speed.
    # We will maintain a set of "satisfied" points.
    # Initially, if S is empty, all points are satisfied (count 0 for all).
    # Adding a point x to S:
    #   - x becomes satisfied if count on some line through x is 1 (itself).
    #   - Other points y on the same lines might become unsatisfied if count becomes > 1 (for S points) or != 0 (for non-S points).
    
    # A faster heuristic:
    # 1. Start with S = empty. All 2401 points satisfied.
    # 2. Try to add points one by one.
    # 3. If adding a point breaks validity for others, we might still add it if it increases total satisfied count?
    #    No, the condition is "Every point must be satisfied". So S must be a valid complement.
    #    This is a hard constraint. We cannot have "partially valid" states in the final output.
    #    However, we can search the space of valid S.
    
    # Let's try a different angle:
    # A set S is valid if for every y, there is a line L through y such that L \cap S is either {y} (if y in S) or empty (if y not in S).
    # This implies that for any y in S, y must be the UNIQUE point of S on at least one line through y.
    # For any y not in S, y must be on at least one line through y that contains NO points of S.
    
    # This looks like finding a large set of "isolated" points in a line geometry.
    
    # Let's use a Random Restart Local Search on the valid set space.
    # But generating a random valid S is hard.
    
    # Alternative: Greedy construction with backtracking?
    # Let's stick to the previous logic but optimize the validation.
    # The previous program's `is_valid` was O(N * D * (N/D)) roughly? No, it used bincount which is O(N).
    # So is_valid is O(N * D). N=2401, D=50. ~120k ops. In numpy, this is very fast.
    # The bottleneck in the previous program was likely the Python loop `for idx in all_points` calling `is_valid` (which is C-speed numpy) but the overhead of 2401 calls * 100 trials = 240k calls.
    # 240k calls * 120k ops = 28 billion ops? No, `is_valid` is one call.
    # Inside `is_valid`, we loop 50 times. Each time `np.bincount` and array ops on 2401 elements.
    # This should be fast enough for 100 trials * 2401 points = 240,100 validations?
    # 240,100 * (50 * small_numpy_ops).
    # Small numpy ops on 2401 elements take ~10-50 microseconds.
    # 50 * 30us = 1.5ms per validation.
    # 240,100 * 1.5ms = 360 seconds. This is too slow!
    
    # We need a faster way to update validity incrementally.
    # Or, we can reduce the number of validations.
    
    # Better Strategy:
    # Instead of adding points one by one and validating the whole set,
    # we can use the structure of the lines.
    
    # Let's try a "Line Packing" approach.
    # If we pick a set of lines to be "empty" (contain no S), then all points on those lines are satisfied (as non-S points).
    # The remaining points must be in S and must be isolated on some other line.
    
    # Actually, the simplest fast heuristic is:
    # 1. Pick a random direction d0.
    # 2. For each line in direction d0, decide if it is "S-only" (contains 1 S point) or "empty" (contains 0 S points).
    #    If a line is "empty", all its points are non-S and satisfied by d0.
    #    If a line is "S-only", we pick one point from it to be in S. That point is satisfied by d0.
    #    The other points on that line are non-S. Are they satisfied?
    #    They are on a line (d0) that contains an S point. So they are NOT satisfied by d0.
    #    They must be satisfied by some OTHER direction d1.
    #    This creates a dependency.
    
    # Let's try a greedy set cover on the "unsatisfied" points.
    # Start with S = empty. All points satisfied.
    # We want to add points to S.
    # When we add x to S:
    #   - x is satisfied if there is a line through x with count 1.
    #   - Points y on lines through x might become unsatisfied if the count on that line becomes 2 (for y not in S) or if y was satisfied by a line that now has count 2.
    
    # Let's use a Randomized Greedy with "Repair".
    # 1. Start with S = empty.
    # 2. Randomly pick a point x.
    # 3. Add x to S.
    # 4. Check validity. If invalid, identify "broken" points.
    # 5. Try to fix by removing some other points or adding points?
    
    # Given the time budget, let's try to optimize the validation loop by using a C-like speed via numpy and reducing the number of full validations.
    # We can batch the checks? No, state changes.
    
    # Let's try a different search:
    # We know that if S is small, it's easy to be valid.
    # We want S to be large.
    # Let's try to construct S by selecting lines.
    
    # New Idea:
    # Use the fact that F_49^2 is a grid.
    # Let's try to find a large independent set in the "line intersection graph" where edges connect points on the same line? No, that's not it.
    
    # Let's go back to the vectorized validation but make it faster by early exiting and using a smarter search.
    # We will perform a "Random Walk" in the space of subsets S.
    # Start with a random S of size k.
    # Compute satisfied points.
    # If not all satisfied, pick an unsatisfied point y.
    # If y is in S, we need to remove some other point from every line through y so that y becomes the only S point on at least one line.
    # If y is not in S, we need to remove all S points from at least one line through y.
    
    # This is a constraint satisfaction problem.
    
    # Let's try a simple heuristic that might work well:
    # 1. Initialize S as empty.
    # 2. While time remains:
    #    a. Pick a random point x not in S.
    #    b. Tentatively add x to S.
    #    c. Check if S is valid.
    #    d. If valid, keep x.
    #    e. If invalid, reject x.
    # This is what the parent did. It was too slow.
    
    # Optimization:
    # Instead of checking validity of the whole set, check only the "affected" points.
    # When adding x, only points on lines through x can change their satisfaction status.
    # There are 50 lines through x. Each has 49 points. Total affected points <= 50*49 = 2450 (almost all).
    # So local check is not much faster than global.
    
    # However, we can maintain the "satisfied" status incrementally.
    # State:
    #   S: set of points.
    #   counts[d, lid]: number of S points on line lid in direction d.
    #   sat[y]: boolean, is y satisfied?
    #   sat_reason[y]: direction d that satisfies y, or -1.
    
    # When adding x to S:
    #   For each direction d:
    #      lid = line_id[x, d]
    #      old_count = counts[d, lid]
    #      new_count = old_count + 1
    #      counts[d, lid] = new_count
    #      
    #      For each point y on line (d, lid):
    #         If y is in S:
    #            y is satisfied by d if new_count == 1.
    #            If new_count > 1, y is NOT satisfied by d.
    #         If y is not in S:
    #            y is satisfied by d if new_count == 0.
    #            If new_count > 0, y is NOT satisfied by d.
            
    #   We need to update sat[y] for all y on these lines.
    #   y is satisfied if ANY direction d satisfies it.
    
    # This incremental update is O(50 * 49) = 2450 operations per addition.
    # 2401 points * 2450 ops = ~6 million ops. This is very fast in Python if optimized, or numpy if vectorized.
    
    # Let's implement this incremental tracker.
    
    # Data structures:
    # S: np.array(bool, n_points)
    # counts: np.array(int, (n_dirs, q))
    # sat: np.array(bool, n_points)
    
    # Precompute line_members[d, lid]: list of points on that line.
    # This is memory heavy: 50 * 49 * 49 = 120k entries. Fine.
    
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            # Find points where line_id[:, d] == lid
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    def initialize_tracker(S):
        counts = np.zeros((n_dirs, q), dtype=np.int32)
        for d in range(n_dirs):
            lids = line_id[S, d]
            counts[d] = np.bincount(lids, minlength=q)
        
        sat = np.zeros(n_points, dtype=bool)
        for y in range(n_points):
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = counts[d, lid]
                if S[y]:
                    if c == 1:
                        sat[y] = True
                        break
                else:
                    if c == 0:
                        sat[y] = True
                        break
        return counts, sat

    def try_add_point(S, counts, sat, x):
        # Returns (new_counts, new_sat, success)
        # x is not in S
        S[x] = True
        
        affected_points = set()
        for d in range(n_dirs):
            lid = line_id[x, d]
            old_c = counts[d, lid]
            new_c = old_c + 1
            counts[d, lid] = new_c
            
            members = line_members[d][lid]
            for y in members:
                affected_points.add(y)
        
        # Update sat for affected points
        # We need to re-evaluate sat for these points
        # Note: points NOT affected might have changed sat if they were relying on a direction where count changed?
        # No, only counts on lines through x changed.
        # So only points on lines through x can change their sat status.
        
        # However, if a point y was satisfied by d1, and d1 is not affected, it remains satisfied.
        # If y was satisfied by d2, and d2 IS affected, it might become unsatisfied.
        # So we only need to check if the point is still satisfied by ANY direction.
        
        for y in affected_points:
            # Check if y is satisfied
            is_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = counts[d, lid]
                if S[y]:
                    if c == 1:
                        is_sat = True
                        break
                else:
                    if c == 0:
                        is_sat = True
                        break
            sat[y] = is_sat
            
        if np.all(sat):
            return True
        else:
            # Revert
            S[x] = False
            for d in range(n_dirs):
                lid = line_id[x, d]
                counts[d, lid] -= 1
            # Revert sat for affected points? 
            # It's easier to just recompute sat for affected points from scratch after revert?
            # Or just return False and let the caller handle state.
            # Since we modified counts and sat in place, we must revert them properly.
            for y in affected_points:
                is_sat = False
                for d in range(n_dirs):
                    lid = line_id[y, d]
                    c = counts[d, lid]
                    if S[y]:
                        if c == 1:
                            is_sat = True
                            break
                    else:
                        if c == 0:
                            is_sat = True
                            break
                sat[y] = is_sat
            return False

    # Main Loop
    start_time = time.time()
    deadline = start_time + 590
    
    best_size = 0
    best_S = None
    
    # Try multiple random orders
    all_points = list(range(n_points))
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        random.shuffle(all_points)
        S = np.zeros(n_points, dtype=bool)
        counts, sat = initialize_tracker(S)
        current_size = 0
        
        for x in all_points:
            if time.time() > deadline:
                break
            if S[x]:
                continue
            if try_add_point(S, counts, sat, x):
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
                    
    # If we didn't find anything, fallback to empty S (N = all points)
    if best_S is None:
        best_S = np.zeros(n_points, dtype=bool)

    # Convert best_S to N
    N_points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        for idx in range(n_points):
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
