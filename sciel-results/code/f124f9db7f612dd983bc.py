import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * 3) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Encode/Decode point index to coordinates
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line IDs and membership masks
    # line_id[i, d] is the ID of the line in direction d passing through point i
    # Line ID is the y-intercept (or x for vertical) encoded as int 0..48
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    coords = np.zeros((n_points, 4), dtype=np.int32)
    
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
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
            # Vertical lines: x = const
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Precompute line members as a list of arrays of point indices
    # line_members[d][lid] is an array of indices of points on that line
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    # --- Max-SAT / Local Search Setup ---
    # Variables: S[i] = True if point i is in the complement (NOT in N)
    # We want to minimize |S| (number of True variables)
    # Constraint for point x: There exists direction d such that NO other point on line(x, d) is in S.
    # Let L be the set of points on line(x, d).
    # Condition: For all y in L \ {x}, S[y] == False.
    # Equivalently: sum(S[y] for y in L \ {x}) == 0.
    
    # For each point x, we have 50 "clauses" (one per direction).
    # A clause is satisfied if the sum of S[y] for y in line(x, d) \ {x} is 0.
    # We want to satisfy ALL clauses for ALL x.
    # If a clause is violated, it contributes to the "cost".
    # Objective: Minimize |S| subject to all clauses satisfied.
    # Since we start with a valid configuration (S=Empty, N=All), we can use a local search
    # that tries to flip S[i] from False to True (remove from N, add to S) if it keeps validity,
    # or more generally, search the space of valid S to find one with min weight?
    # Wait, the current program minimizes N (maximizes S).
    # Let's stick to that: Maximize |S| (complement) such that S is an "independent set" in the line hypergraph.
    # Actually, the condition is: For every x, there is a line L through x such that L \ {x} subset N.
    # This is equivalent to: For every x, there is a line L through x such that L \ {x} intersect S = Empty.
    # So, if x is in S, we need a line L through x where all other points are in N (not in S).
    # If x is in N, we need a line L through x where all other points are in N (not in S).
    # In both cases, the requirement is: For every x, exists d such that (Line(x, d) \ {x}) intersect S = Empty.
    
    # This means S cannot contain any two points that "conflict"?
    # No, it's a global constraint.
    # Let's define Cost(S) = number of points x for which NO direction d satisfies the condition.
    # We want Cost(S) = 0 and |S| maximized.
    
    # Initial state: S = Empty (all False). Cost = 0. |S| = 0.
    # We want to add points to S (set to True) one by one, or use random restarts.
    # Adding a point x to S might break the condition for other points y that relied on lines containing x being "clean" (S-free).
    
    # Let's use a Randomized Greedy / WalkSAT approach.
    # 1. Start with S = Empty.
    # 2. Try to add points to S in a random order.
    # 3. When adding x, check if it breaks validity for any point.
    #    If it does, we have a conflict.
    #    Instead of simple backtracking, we can use a "repair" strategy or just skip x.
    #    But skipping might lead to suboptimal |S|.
    
    # Better: Simulated Annealing or WalkSAT on the "Cost" function.
    # State: S (boolean array of size 2401)
    # Cost: Number of "unsatisfied" points x.
    #    A point x is satisfied if there exists d such that no other point on line(x,d) is in S.
    #    Let valid[x] = True if x is satisfied.
    #    Cost = sum(~valid).
    #    We want Cost = 0.
    #    Among Cost=0 states, maximize |S|.
    
    # Efficient update:
    # When we flip S[x] (add or remove x from S), how does it affect validity of other points?
    # x affects validity of any point y such that x is on a line through y.
    # i.e., for any direction d, if y is on a line with x in direction d, then y's validity in direction d might change.
    # Specifically, if S[x] becomes True, then for any y such that x is on line(y, d), the clause (line(y,d)\{y}) intersect S is no longer empty if x != y.
    # So y loses satisfaction in direction d.
    # If y had ONLY that direction d satisfying it, y becomes unsatisfied.
    
    # Data structures for fast update:
    # For each point y, and each direction d, we need to know if line(y, d) \ {y} contains any point in S.
    # Let clean[y, d] = True if line(y, d) \ {y} is disjoint from S.
    # y is satisfied if any clean[y, d] is True.
    
    # Precompute for each point x, which (y, d) pairs are affected if S[x] changes.
    # If S[x] changes, it affects clean[y, d] for all y, d such that x is on line(y, d) and x != y.
    # This is equivalent to: for all y, and for all d such that x is on line(y, d).
    # Note: For a fixed x and y, there is exactly one direction d (unless x=y) such that x is on line(y, d).
    # So for each x, it affects clean[y, d_xy] for all y != x.
    # Where d_xy is the direction of the line passing through x and y.
    
    # Let's precompute `affected_by[x]` = list of (y, d) pairs?
    # Size: 2401 * 2400 = ~5.7 million pairs. This is manageable.
    # Or, for each x, we can just iterate through all y? No, that's slow.
    # We can precompute for each x, the list of (y, d) that x influences.
    
    # Actually, let's look at the structure.
    # For a fixed x, and any y != x, there is a unique line through x and y.
    # This line has a direction d.
    # So x influences clean[y, d].
    # We can precompute a list `influences[x]` of tuples (y, d).
    # Length of each list is 2400. Total 2401 * 2400 ~ 5.7M entries.
    # Memory: 5.7M * 2 ints ~ 45 MB. Acceptable.
    
    # Let's precompute `influences`.
    # influences[x] = [(y, d_xy) for y in range(n_points) if y != x]
    
    # To compute d_xy:
    # We need the direction of the line passing through x and y.
    # We can use the precomputed line_id.
    # For a given x, and a direction d, the line is determined.
    # We can build a map: for each x, and each d, the line ID is line_id[x, d].
    # All points y on that line have line_id[y, d] == line_id[x, d].
    # So, for a fixed x, and direction d, the points y (y != x) on the same line in direction d are those where line_id[y, d] == line_id[x, d].
    # These y are affected by S[x] in direction d.
    
    # So, for each x, we can iterate d from 0 to 49.
    # For each d, find all y such that line_id[y, d] == line_id[x, d] and y != x.
    # These are the points y whose clean[y, d] depends on S[x].
    
    # Let's build `influences[x]` as a list of (y, d).
    # We can do this efficiently using the line_members structure.
    # For each x:
    #   for d in range(n_dirs):
    #       lid = line_id[x, d]
    #       members = line_members[d][lid]
    #       for y in members:
    #           if y != x:
    #               influences[x].append((y, d))
    
    # This is 2401 * 50 * 49 ~ 5.8M operations. Fast enough.
    
    influences = [None] * n_points
    for x in range(n_points):
        infl = []
        for d in range(n_dirs):
            lid = line_id[x, d]
            members = line_members[d][lid]
            for y in members:
                if y != x:
                    infl.append((y, d))
        influences[x] = infl

    # State:
    # S: boolean array, S[i] = True if i in complement.
    # clean[y, d]: boolean, True if line(y, d) \ {y} is disjoint from S.
    # sat_count[y]: number of directions d such that clean[y, d] is True.
    # valid[y]: sat_count[y] > 0.
    # cost: number of y such that valid[y] is False.
    # size_S: sum(S).
    
    # Initialization: S = all False.
    # Then clean[y, d] is True for all y, d (since S is empty, intersection is empty).
    # sat_count[y] = 50 for all y.
    # valid[y] = True for all y.
    # cost = 0.
    # size_S = 0.
    
    S = np.zeros(n_points, dtype=bool)
    clean = np.ones((n_points, n_dirs), dtype=bool)
    sat_count = np.full(n_points, n_dirs, dtype=np.int32)
    valid = np.ones(n_points, dtype=bool)
    cost = 0
    size_S = 0
    
    best_S = S.copy()
    best_size = 0
    
    # Local Search:
    # We want to increase size_S while keeping cost = 0.
    # Move: Flip S[x] from False to True (add to S).
    # Check if cost remains 0.
    # If yes, accept.
    # If no, reject (or try to repair).
    
    # To allow exploration, we can also allow moves that increase cost temporarily?
    # No, let's stick to valid moves first. If stuck, we can try to remove points from S to unblock?
    # Or use Simulated Annealing on (cost, -size_S).
    
    # Let's try a simple Greedy + Random Restart strategy.
    # 1. Start with S = Empty.
    # 2. While time < deadline:
    #      Pick a random point x with S[x] == False.
    #      Tentatively set S[x] = True.
    #      Update clean, sat_count, valid, cost.
    #      If cost == 0:
    #          Accept. Update best.
    #      Else:
    #          Reject. Revert.
    #      If stuck (no more points can be added), try to remove a random point from S to create space?
    #      Or just restart with a different random order?
    
    # To make it more powerful, let's use a "WalkSAT" style repair if we get stuck?
    # Or just multiple random restarts.
    
    # Optimization:
    # When flipping S[x] to True, we need to update clean[y, d] for all (y, d) in influences[x].
    # For each (y, d), if S[x] becomes True, then clean[y, d] becomes False ONLY IF x was the ONLY point in S on that line?
    # No, clean[y, d] is "disjoint from S".
    # If S[x] was False and becomes True, then x is now in S.
    # So clean[y, d] becomes False. (Because x is in line(y, d) \ {y} and x is in S).
    # Wait, is x in line(y, d) \ {y}? Yes, by construction of influences.
    # So clean[y, d] MUST become False.
    # It doesn't matter if other points are in S. If x is in S, the intersection is not empty.
    
    # So, flipping S[x] from False to True:
    # For each (y, d) in influences[x]:
    #     if clean[y, d] was True:
    #         clean[y, d] = False
    #         sat_count[y] -= 1
    #         if sat_count[y] == 0:
    #             valid[y] = False
    #             cost += 1
    #     else:
    #         # clean[y, d] was already False, so it stays False. No change to sat_count.
    #         pass
    
    # Flipping S[x] from True to False:
    # For each (y, d) in influences[x]:
    #     if clean[y, d] was False:
    #         # We need to check if there are OTHER points in S on line(y, d) \ {y}.
    #         # If no other points in S, then clean[y, d] becomes True.
    #         # How to check?
    #         # We need to know the count of points in S on line(y, d) \ {y}.
    #         # Let's maintain count_S[y, d] = number of points z in S such that z is on line(y, d) and z != y.
    #         # Then clean[y, d] is True iff count_S[y, d] == 0.
        
    # Let's add count_S array.
    # count_S[y, d] = sum(S[z] for z in line(y, d) if z != y).
    
    # Update count_S:
    # When S[x] changes, it affects count_S[y, d] for all (y, d) in influences[x].
    # If S[x] False -> True: count_S[y, d] += 1
    # If S[x] True -> False: count_S[y, d] -= 1
    
    # Then clean[y, d] = (count_S[y, d] == 0).
    # And sat_count[y] = sum(clean[y, d] for d).
    # valid[y] = (sat_count[y] > 0).
    
    # Let's re-implement with count_S.
    
    S = np.zeros(n_points, dtype=bool)
    count_S = np.zeros((n_points, n_dirs), dtype=np.int32)
    sat_count = np.full(n_points, n_dirs, dtype=np.int32)
    valid = np.ones(n_points, dtype=bool)
    cost = 0
    size_S = 0
    
    best_S = S.copy()
    best_size = 0
    
    # Helper to apply a flip
    def flip(x, to_val):
        nonlocal cost, size_S
        current_val = S[x]
        if current_val == to_val:
            return
        
        S[x] = to_val
        if to_val:
            size_S += 1
        else:
            size_S -= 1
            
        for y, d in influences[x]:
            if to_val:
                count_S[y, d] += 1
                if count_S[y, d] == 1:
                    # Was 0, now 1. clean becomes False.
                    sat_count[y] -= 1
                    if sat_count[y] == 0:
                        valid[y] = False
                        cost += 1
            else:
                count_S[y, d] -= 1
                if count_S[y, d] == 0:
                    # Was 1, now 0. clean becomes True.
                    sat_count[y] += 1
                    if sat_count[y] == 1:
                        valid[y] = True
                        cost -= 1

    # Search Loop
    # We want to maximize size_S with cost=0.
    # Strategy:
    # 1. Greedy Add: Try to add random points.
    # 2. If stuck, try to remove a point that is "critical" to allow adding others?
    #    Or just restart.
    
    # Let's do multiple random restarts.
    # In each restart, we try to build S from scratch using a random order of points.
    # If we get stuck, we stop the restart.
    
    n_restarts = 200 # Adjust based on time
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        S[:] = False
        count_S[:] = 0
        sat_count[:] = n_dirs
        valid[:] = True
        cost = 0
        size_S = 0
        
        # Random order of points
        points = list(range(n_points))
        random.shuffle(points)
        
        stuck = False
        for x in points:
            if time.time() > deadline:
                break
            if S[x]:
                continue
            # Try to add x
            flip(x, True)
            if cost == 0:
                # Accept
                if size_S > best_size:
                    best_size = size_S
                    best_S = S.copy()
            else:
                # Reject
                flip(x, False)
        
        # If we finished the loop, we might be stuck (no more points could be added).
        # We could try to remove some points to unblock, but for now, let's just restart.
        # To improve, we can try a "repair" phase:
        # If cost > 0, we must have rejected points.
        # Actually, we only accept if cost==0. So cost is always 0 in this greedy approach.
        # The problem is we might miss a better combination.
        
        # Let's add a local improvement phase:
        # Try to swap: Remove one point, add another?
        # Or just try to add points that were previously rejected, now that the set is different?
        # No, the set is the same.
        
        # Let's try a second pass in reverse order?
        # Or just rely on multiple restarts.
        
    # Final result
    N = ~best_S
    
    # Convert to array
    N_points = []
    for idx in range(n_points):
        if N[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        N_points = [[[0,0],[0,0]]]
        
    return np.array(N_points, dtype=np.int64)

nikodym_set()
