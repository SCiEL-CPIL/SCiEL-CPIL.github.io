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
    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

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

    # Precompute line_members for incremental updates
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    # Validation function using vectorized numpy
    def is_valid(S):
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            counts = np.bincount(lids[S], minlength=q)
            counts_for_points = counts[lids]
            valid_for_d = (S & (counts_for_points == 1)) | (~S & (counts_for_points == 0))
            satisfied |= valid_for_d
            if np.all(satisfied):
                return True
        return False

    def count_satisfied(S):
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            counts = np.bincount(lids[S], minlength=q)
            counts_for_points = counts[lids]
            valid_for_d = (S & (counts_for_points == 1)) | (~S & (counts_for_points == 0))
            satisfied |= valid_for_d
        return np.sum(satisfied)

    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    best_S = None
    best_complement = 0
    
    # Strategy: Greedy construction with local search
    # We want to maximize the complement size = 2401 - |S|
    # So we want to minimize |S| while keeping S valid.
    
    # But S must satisfy: for every point y, exists direction d such that:
    # - if y in S: count of S on line(y,d) == 1
    # - if y not in S: count of S on line(y,d) == 0
    
    # Let's try to find a small valid S.
    # Start with S empty. All points satisfied (count 0 for all non-S points).
    # We want to REMOVE points from the complement (add to S) only if necessary?
    # No, we want the complement to be LARGE. So S should be SMALL.
    # The minimum S is size 0? If S is empty, all points are non-S, and for each point y, 
    # we need a line through y with 0 S points. Since S is empty, every line has 0 S points.
    # So S = empty is valid! The complement is all 2401 points.
    # But the metric is 2401 - |N|, where N is the returned set.
    # Wait, the task says: "returning ... listing the points of a set N".
    # "Metric: the complement size 2401 - |N|".
    # So if N is the Nikodym set, and we want to MAXIMIZE 2401 - |N|, we want |N| to be MINIMAL.
    # If N can be empty, then 2401 - 0 = 2401 is the max.
    # But is N=empty valid? 
    # "N must be a Nikodym set: every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N."
    # If x is not in N (which is all points), the line must have all other points in N.
    # If N is empty, the line has no points in N. So "points other than x all belong to N" is vacuously true?
    # No, "all belong to N" means the set of other points is a subset of N.
    # If N is empty, the other points are not in N. So it's false.
    # So N cannot be empty.
    
    # Let's re-read carefully.
    # "every point x ... must lie on a line ... whose points other than x all belong to N."
    # If x in N: other points on line must be in N.
    # If x not in N: other points on line must be in N.
    # This implies that for any x, there is a line through x such that (Line \ {x}) subset N.
    
    # If N is the set of points we return.
    # We want to MAXIMIZE 2401 - |N|. So we want |N| to be as small as possible.
    
    # Let's test S = complement of N.
    # The condition is about N.
    # Let's stick to constructing N directly.
    
    # Actually, the previous code constructed S (the complement of N) and tried to maximize |S|?
    # "Metric (higher is better): the complement size 2401 - |N|".
    # The code returns N_points which are the points NOT in S.
    # So N = complement of S.
    # So |N| = 2401 - |S|.
    # Metric = 2401 - |N| = 2401 - (2401 - |S|) = |S|.
    # So the metric is |S|. We want to MAXIMIZE |S|.
    
    # So we want S to be as large as possible.
    # S is the set of points NOT in N.
    # Condition for N: for every x, exists line L through x such that (L \ {x}) subset N.
    # Since N = All \ S, (L \ {x}) subset N means (L \ {x}) is disjoint from S.
    # So for every x, there is a line L through x such that L \ {x} contains NO points from S.
    
    # If x in S: L \ {x} disjoint from S. Since x in S, this means L intersects S only at x (if any other point in L was in S, it would be in L\{x}).
    # So if x in S, x must be the ONLY point of S on L.
    # If x not in S: L \ {x} disjoint from S. Since x not in S, this means L contains NO points from S.
    
    # This matches the previous logic:
    # S is valid if:
    # - For x in S: exists d such that count(S on line(x,d)) == 1
    # - For x not in S: exists d such that count(S on line(x,d)) == 0
    
    # So we want to MAXIMIZE |S| subject to this constraint.
    
    # The previous code tried to add points to S greedily.
    # Let's improve the search.
    
    # We can use a local search (simulated annealing or hill climbing) to maximize |S|.
    # Start with a random valid S (e.g., empty, or small random set that is valid).
    # Then try to add points. If adding a point breaks validity, try to repair by removing other points.
    
    # Better: Start with empty S (valid, size 0).
    # Greedily add points that keep S valid.
    # This might get stuck in a local optimum.
    # Use random restarts and local search (add/remove/swap points).
    
    # Let's implement a fast local search.
    
    def try_add_point_fast(S, counts, sat, x):
        # Returns (success, new_counts, new_sat) or (False, None, None)
        # x is not in S
        # Check if adding x keeps S valid.
        
        # Tentatively add x
        # Check if x is satisfied (it should be, if any line through x has count 1 after adding)
        # Actually, we just need to check if the resulting S is valid.
        # For speed, we only check affected points.
        
        # Affected points: all points on lines through x.
        affected = set()
        for d in range(n_dirs):
            lid = line_id[x, d]
            affected.update(line_members[d][lid].tolist())
        
        # Check validity for affected points
        # We need to check if they are still satisfied.
        # A point y is satisfied if:
        # - y in S: count on some line through y is 1
        # - y not in S: count on some line through y is 0
        
        # For points not affected, their status doesn't change.
        # For affected points, we recompute their satisfaction.
        
        # We need the counts AFTER adding x.
        # Let's compute the new counts for lines through x.
        
        # Note: counts is a 2D array (n_dirs, q).
        # We don't want to modify it in place yet, or we revert if it fails.
        
        # Let's just check validity of the whole S with x added?
        # is_valid is fast enough?
        # is_valid is O(N * D) ~ 120k ops.
        # If we do this for every candidate x, it's slow.
        
        # Let's stick to the incremental check.
        
        # For each affected point y:
        #   Is y satisfied in the new state?
        
        new_counts = counts.copy()
        for d in range(n_dirs):
            lid = line_id[x, d]
            new_counts[d, lid] += 1
            
        # Check x
        x_sat = False
        for d in range(n_dirs):
            lid = line_id[x, d]
            if new_counts[d, lid] == 1:
                x_sat = True
                break
        if not x_sat:
            return False, None, None
            
        # Check other affected points
        for y in affected:
            if y == x:
                continue
            # y's satisfaction depends on its new status (S[y] unchanged, but counts changed)
            y_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = new_counts[d, lid]
                if S[y]:
                    if c == 1:
                        y_sat = True
                        break
                else:
                    if c == 0:
                        y_sat = True
                        break
            if not y_sat:
                return False, None, None
                
        # If all affected points are satisfied, then the whole S is valid.
        # (Unaffected points were satisfied before and remain so).
        
        # Update sat for affected points
        new_sat = sat.copy()
        for y in affected:
            y_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = new_counts[d, lid]
                if S[y]:
                    if c == 1:
                        y_sat = True
                        break
                else:
                    if c == 0:
                        y_sat = True
                        break
            new_sat[y] = y_sat
            
        return True, new_counts, new_sat

    def try_remove_point_fast(S, counts, sat, x):
        # x is in S
        # Remove x from S.
        # Check if S is still valid.
        
        affected = set()
        for d in range(n_dirs):
            lid = line_id[x, d]
            affected.update(line_members[d][lid].tolist())
            
        new_counts = counts.copy()
        for d in range(n_dirs):
            lid = line_id[x, d]
            new_counts[d, lid] -= 1
            
        # Check affected points
        for y in affected:
            y_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = new_counts[d, lid]
                if S[y] and y != x: # x is removed, so not in S anymore
                    if c == 1:
                        y_sat = True
                        break
                else: # y not in S (including x)
                    if c == 0:
                        y_sat = True
                        break
            if not y_sat:
                return False, None, None
                
        # Update sat
        new_sat = sat.copy()
        for y in affected:
            y_sat = False
            for d in range(n_dirs):
                lid = line_id[y, d]
                c = new_counts[d, lid]
                if S[y] and y != x:
                    if c == 1:
                        y_sat = True
                        break
                else:
                    if c == 0:
                        y_sat = True
                        break
            new_sat[y] = y_sat
            
        return True, new_counts, new_sat

    # Main search loop
    # We will perform multiple random restarts.
    # In each restart, we start with empty S and greedily add points.
    # Then we do local search: try to add more points, or swap points.
    
    all_points = list(range(n_points))
    
    for trial in range(100):
        if time.time() > deadline:
            break
            
        random.shuffle(all_points)
        S = np.zeros(n_points, dtype=bool)
        
        # Initialize counts and sat
        counts = np.zeros((n_dirs, q), dtype=np.int32)
        sat = np.ones(n_points, dtype=bool) # Empty S is valid, all satisfied
        
        current_size = 0
        
        # Greedy add
        for x in all_points:
            if time.time() > deadline:
                break
            if S[x]:
                continue
            success, new_counts, new_sat = try_add_point_fast(S, counts, sat, x)
            if success:
                S[x] = True
                counts = new_counts
                sat = new_sat
                current_size += 1
                
        # Local search
        # Try to add points that were rejected?
        # Or try to improve by swapping?
        # For simplicity, let's just record the result.
        
        if current_size > best_complement:
            best_complement = current_size
            best_S = S.copy()
            
        if time.time() > deadline:
            break
            
    if best_S is None:
        best_S = np.zeros(n_points, dtype=bool)
        
    # Construct N
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
