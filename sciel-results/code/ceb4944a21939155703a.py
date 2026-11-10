# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    Strategy: 
    1. Represent S (complement of N) as a boolean array.
    2. Start with S = all points (invalid).
    3. Greedily remove points from S to make B(S) empty (where B(S) are points x 
       for which all lines through x intersect S \ {x}).
    4. Once valid, try to add points back to S to maximize |S|, ensuring validity is maintained.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 550  # safety margin
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3
    
    # Precompute line structures
    # directions: v = (1, m) for m in F_q, and v = (0, 1)
    # There are q + 1 = 50 directions.
    # For each point x and direction v, we need to know which other points are on the line.
    # line_members[point_idx][v_idx] = array of point indices on the line, excluding point_idx.
    
    # Directions
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    num_dirs = len(directions)
    
    # Precompute line members for each point and direction
    # This is 2401 * 50 * 48 integers. 2401*50*48 ~ 5.7M ints. ~23MB. Acceptable.
    line_members = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            point_idx = x_idx * q + y_idx
            
            for v_idx, (dx_idx, dy_idx) in enumerate(directions):
                dx_a, dx_b = divmod(dx_idx, p)
                dy_a, dy_b = divmod(dy_idx, p)
                
                count = 0
                for t in range(q):
                    if t == 0:
                        continue
                    t_a, t_b = divmod(t, p)
                    
                    # t * dx
                    tdx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                    tdx_b = (t_a * dx_b + t_b * dx_a) % p
                    tdx_idx = tdx_a * p + tdx_b
                    
                    # t * dy
                    tdy_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    tdy_b = (t_a * dy_b + t_b * dy_a) % p
                    tdy_idx = tdy_a * p + tdy_b
                    
                    # x + t*v
                    nx_a = (x_a + tdx_a) % p
                    nx_b = (x_b + tdx_b) % p
                    ny_a = (y_a + tdy_a) % p
                    ny_b = (y_b + tdy_b) % p
                    
                    n_point_idx = nx_a * p * q + nx_b * q + ny_a * p + ny_b
                    
                    line_members[point_idx, v_idx, count] = n_point_idx
                    count += 1
    
    # S is the set of points NOT in N.
    # We want to maximize |S| such that for all x, exists v with L(x, v) \ {x} \cap S = empty.
    
    # Initialize S as all True (all points in S, N is empty)
    S = np.ones(total_points, dtype=bool)
    
    # Function to compute B(S): points x such that for ALL v, L(x, v) \ {x} \cap S is not empty.
    # bad_count[x] = number of directions v for which L(x, v) \ {x} \cap S is NOT empty.
    # If bad_count[x] == num_dirs, then x is in B(S).
    
    def compute_bad_counts(S_arr):
        bad_counts = np.zeros(total_points, dtype=np.int32)
        for x_idx in range(total_points):
            # For each direction, check if any point in line is in S
            for v_idx in range(num_dirs):
                # line_members[x_idx, v_idx] contains indices of points on the line
                # We need to check if any of them are in S_arr
                # np.any(S_arr[line_members[x_idx, v_idx]])
                if np.any(S_arr[line_members[x_idx, v_idx]]):
                    bad_counts[x_idx] += 1
        return bad_counts
    
    # Greedy Shrinkage: Remove points from S to make B(S) empty.
    # A point p is "bad" if it contributes to making some x in B(S).
    # Actually, we want to remove points to DECREASE bad_counts of points in B(S) to < num_dirs.
    
    # Initial bad counts
    bad_counts = compute_bad_counts(S)
    num_bad = np.sum(bad_counts == num_dirs)
    
    # Helper to get "utility" of removing point p.
    # Removing p from S will decrease bad_counts[x] by 1 for all x such that 
    # p is in L(x, v) \ {x} AND L(x, v) \ {x} \cap S == {p} (i.e., p was the ONLY point in S on that line).
    # If bad_counts[x] was num_dirs, and it becomes num_dirs - 1, x is no longer in B(S).
    # So utility is the number of x in B(S) for which p is the unique blocker in at least one direction.
    # More precisely:
    # For a fixed x, let D(x) be the set of directions v such that L(x, v) \ {x} \cap S is NOT empty.
    # x is in B(S) iff |D(x)| == num_dirs.
    # If we remove p, D(x) might lose some directions.
    # Specifically, for direction v, if L(x, v) \ {x} \cap S == {p}, then v is removed from D(x).
    # So removing p reduces |D(x)| by the number of directions v where p is the unique element of S on L(x, v) \ {x}.
    # Let unique_block_count[x, p] be this number.
    # If x is in B(S) (|D(x)| == num_dirs), and unique_block_count[x, p] > 0, then x leaves B(S).
    # Utility of p = sum over x in B(S) of (1 if unique_block_count[x, p] > 0 else 0).
    
    # Precomputing unique_block_count for all x, p is expensive (2401^2 * 50).
    # Instead, we can compute it on the fly for candidate points or use a heuristic.
    
    # Heuristic: Randomly pick a point from S to remove, prioritizing points that are in many lines of bad points.
    # Or, just iterate through points and estimate impact.
    
    # Let's use a simpler greedy:
    # While B(S) is not empty:
    #   Find a point p in S that is "most critical".
    #   A point p is critical if it lies on lines L(x, v) where x is bad and v is "tight" (only p blocks it).
    
    # To speed this up, we can maintain a list of bad points.
    # And for each bad point, find a direction v where it is blocked.
    # Then pick a point p on that line to remove.
    
    # Let's try a different approach: Local Search from a valid set.
    # Start with S = empty (Valid).
    # Try to add points.
    # This is what the current program does.
    
    # The directive asks for a STRUCTURAL change.
    # Let's try: **Simulated Annealing on S**.
    # Start with S = empty.
    # Current state: S.
    # Validity: B(S) == empty.
    # Objective: Maximize |S|.
    # Move: Pick a point p not in S, add it. If valid, keep. If invalid, maybe keep with prob exp(-delta/T)?
    # But we must end with a valid set.
    # So, we can do:
    # 1. Start with S = empty.
    # 2. While time < deadline:
    #    a. Try to add a random point p to S.
    #    b. Check if S \cup {p} is valid.
    #    c. If valid, accept.
    #    d. If invalid, reject.
    #    e. Also, try to swap: remove p1 from S, add p2.
    
    # This is essentially the same as greedy growth but with randomization and swaps.
    # Is it materially different?
    # The current program does greedy growth: "For each point not in S, check if it can be added".
    # This is deterministic.
    # Randomized search (Simulated Annealing) explores different local optima.
    
    # Let's implement Simulated Annealing.
    
    S = np.zeros(total_points, dtype=bool)
    
    # Precompute line_members is already done.
    
    # Function to check validity of S
    def is_valid(S_arr):
        # For all x, exists v such that L(x, v) \ {x} \cap S is empty.
        # Equivalent: For all x, bad_count[x] < num_dirs.
        # We can compute bad_counts efficiently.
        bad_counts = np.zeros(total_points, dtype=np.int32)
        for x_idx in range(total_points):
            for v_idx in range(num_dirs):
                if np.any(S_arr[line_members[x_idx, v_idx]]):
                    bad_counts[x_idx] += 1
                    if bad_counts[x_idx] == num_dirs:
                        return False
        return True
    
    # To speed up is_valid, we can maintain bad_counts incrementally.
    # But for simplicity, let's use the full check if the set is small.
    # As S grows, is_valid becomes slower.
    
    # Better: Maintain bad_counts.
    bad_counts = np.zeros(total_points, dtype=np.int32)
    
    def update_bad_counts_on_add(S_arr, p):
        # p is added to S.
        # For each x, for each v, if p is in L(x, v) \ {x}, then bad_counts[x] might increase.
        # Specifically, if L(x, v) \ {x} \cap S was empty before, it is not empty now.
        # So bad_counts[x] increases by 1 for each (x, v) such that p in L(x, v) \ {x} and 
        # (L(x, v) \ {x} \cap S) was empty.
        # How to find such (x, v)?
        # p is in L(x, v) \ {x} iff x is in L(p, v) \ {p}.
        # So for each v, for each x in L(p, v) \ {p}, check if L(x, v) \ {x} \cap S was empty.
        # This is still O(q^2) per addition.
        
        # Let's just recompute bad_counts from scratch if S is small, or use a faster check.
        pass
    
    # Given the time limit, let's stick to a robust Greedy with Randomization and Swaps.
    
    # 1. Initialize S with a small random valid set?
    #    S = empty is valid.
    
    # 2. Greedy Add:
    #    Try to add points that are "easy" to add.
    #    A point p is easy if it doesn't block many lines.
    #    Actually, adding p blocks lines L(x, v) where p in L(x, v) \ {x}.
    #    We want p to block lines that are ALREADY blocked (so bad_counts doesn't increase for bad points)
    #    or lines that are not critical.
    
    # Let's try a different structural approach: **Constraint Programming / Backtracking**.
    # No, too slow.
    
    # Let's go back to **Greedy Shrinkage from Full Set**.
    # It guarantees a valid set quickly.
    # Then we try to **Add Back** points.
    
    S = np.ones(total_points, dtype=bool)
    bad_counts = compute_bad_counts(S)
    num_bad = np.sum(bad_counts == num_dirs)
    
    # Shrinkage Phase
    while num_bad > 0 and time.time() < deadline - 100:
        # Find a point p to remove.
        # Heuristic: Pick a point p that is in S and lies on a line L(x, v) where x is bad and 
        # v is a direction where L(x, v) \ {x} \cap S is "small" (few points in S).
        # Removing p from a small set is more likely to unblock x.
        
        # Find a bad point x.
        bad_x = np.where(bad_counts == num_dirs)[0]
        if len(bad_x) == 0:
            break
        x = bad_x[0]
        
        # Find a direction v for x where L(x, v) \ {x} \cap S is not empty.
        # Pick the one with the fewest points in S.
        min_count = q
        best_v = -1
        best_p = -1
        for v_idx in range(num_dirs):
            line_pts = line_members[x, v_idx]
            # Count points in S
            cnt = np.sum(S[line_pts])
            if cnt > 0 and cnt < min_count:
                min_count = cnt
                best_v = v_idx
                # Pick a point p from this line to remove
                # Prefer one that is in S
                s_pts = line_pts[S[line_pts]]
                if len(s_pts) > 0:
                    best_p = s_pts[0]
        
        if best_p != -1:
            S[best_p] = False
            # Recompute bad_counts (expensive, but okay for now)
            bad_counts = compute_bad_counts(S)
            num_bad = np.sum(bad_counts == num_dirs)
        else:
            # Stuck? Should not happen if S is not empty.
            break
    
    # Now S is valid.
    # Growth Phase: Try to add points back.
    # Try adding points that were removed, or points not in S.
    # Actually, S is the complement of N. We want to maximize |S|.
    # So we want to ADD points to S.
    
    # List of candidates: all points not in S.
    candidates = np.where(~S)[0]
    random.shuffle(candidates)
    
    for p in candidates:
        if time.time() > deadline - 10:
            break
        
        # Try adding p to S
        S[p] = True
        # Check validity
        # We can check if B(S) is empty.
        # To speed up, check if adding p makes any x bad.
        # x becomes bad if bad_counts[x] becomes num_dirs.
        # bad_counts[x] increases if p is in L(x, v) \ {x} and L(x, v) \ {x} \cap S was empty.
        
        # Let's just recompute bad_counts for now.
        bad_counts_new = compute_bad_counts(S)
        if np.any(bad_counts_new == num_dirs):
            S[p] = False
        else:
            # Keep p in S
            pass
    
    # Build result
    points = []
    for point_idx in range(total_points):
        if not S[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    if not points:
        points = [[[0, 0], [0, 0]]]
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
