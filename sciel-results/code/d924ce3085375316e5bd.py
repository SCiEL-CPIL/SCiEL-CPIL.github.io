import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# For Min-Conflicts, we need to quickly find if a set S is valid.
# A set S is invalid if there exist a, b, c in S such that:
# 1. d(a,c) == d(b,c) (a, b equidistant from c)
# 2. d(a,c) == d(a,b) (a, c equidistant from b? No, d(a,c)=d(a,b) means b,c equidistant from a)
# Generally: for any distinct a,b,c in S, no two of d(a,b), d(a,c), d(b,c) are equal.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_size = 0
    best_set = []
    
    # We will use a Min-Conflicts style local search.
    # State: A boolean array `in_set` and a list `S` of indices in the set.
    # Objective: Minimize number of conflicts (isosceles triangles).
    # Move: Remove a point involved in many conflicts, then try to add a new point.
    
    # To make this fast, we need to estimate conflicts.
    # Full conflict check is O(|S|^3) or O(|S|^2 log |S|) with sorting.
    # We can approximate or use a fast "potential conflict" check.
    
    # Actually, let's stick to a robust "Add if Valid" but with a smarter restart strategy 
    # and a "shake" move that removes multiple points to escape local optima.
    # The current program's SA was slow because `is_valid_addition` was O(|S|^2) with high constant.
    
    # Optimization: Precompute distance ranks? No.
    # Let's use the "Forbidden Distance" bitmask idea from inspiration 1 more directly.
    # For a candidate point k, it is invalid if:
    # 1. There exist a, b in S such that d(k,a) == d(k,b).
    # 2. There exist a, b in S such that d(k,a) == d(a,b).
    
    # We can precompute for each k, the set of distances to all other points.
    # But we need to check against the *current* S.
    
    # Let's try a different structural approach: **Iterative Repair**.
    # 1. Start with a random subset of size M (e.g., 1.5x expected best).
    # 2. While conflicts exist:
    #    a. Find a conflict (a,b,c).
    #    b. Remove one of them (e.g., random, or the one with most conflicts).
    # 3. Try to add random points until no more can be added.
    
    # To find conflicts fast:
    # For each pair (a,b) in S, check if any c in S has d(a,c)==d(b,c) or d(a,c)==d(a,b) or d(b,c)==d(a,b).
    # This is still heavy.
    
    # Let's optimize the validity check for a single point addition, which is the core of most heuristics.
    # We will maintain a list S.
    # `is_valid_addition(k)`:
    # 1. `dks = DIST[k, S_arr]`
    # 2. If duplicates in dks -> Invalid.
    # 3. For each a in S, if d(k,a) is in `DIST[a, S_arr]` -> Invalid.
    
    # The bottleneck in step 3 is the inner loop.
    # We can vectorize step 3 partially?
    # We have `dks` (vector of length m).
    # We have `DIST[S_arr, S_arr]` (matrix m x m).
    # We want to know if for any i, `dks[i]` is in `DIST[S_arr[i], :]`.
    # This is a row-wise membership check.
    
    # Alternative: **Greedy with Backtracking / Branch and Bound**? No, too slow.
    
    # Let's try **Simulated Annealing** but with a **much faster validity check** 
    # by precomputing "distance sets" for points in S?
    # No, S changes.
    
    # What if we use **Bitsets** for distances?
    # Max squared distance is (63^2 + 63^2) = 7938.
    # We can represent the set of distances from a point k to S as a bitset of size ~8000.
    # Python ints are arbitrary precision, so we can use an int as a bitset!
    
    # `dist_bits[k]` would be an int where bit `d` is 1 if distance `d` exists in the set of distances from k to S.
    # But this depends on S.
    
    # However, we can maintain for each point `p` in S, a bitset `B[p]` of distances from p to *all other points in S*.
    # When we add a point k:
    # 1. For each p in S, check if `DIST[k, p]` is already in `B[p]`. If yes, conflict.
    # 2. Also check if `DIST[k, p]` is in `B[k]` (which is built as we go)?
    #    Actually, condition 1 covers "d(k,p) == d(p,q)".
    #    Condition 2 is "d(k,p) == d(k,q)". This is equivalent to checking if `DIST[k, p]` is in the set of distances from k to other points in S.
    
    # Let's maintain:
    # `S`: list of indices.
    # `B`: list of Python ints (bitsets) corresponding to points in S.
    # `B[i]` has bit `d` set if there exists `j != i` in S such that `DIST[S[i], S[j]] == d`.
    
    # Adding k:
    # 1. Create `new_B_k` = 0.
    # 2. For each i, p in S:
    #    d = DIST[k, p]
    #    If `B[i]` has bit `d` set: Conflict (d(k,p) == d(p,q) for some q in S).
    #    If `new_B_k` has bit `d` set: Conflict (d(k,p) == d(k,q) for some q in S).
    #    Set bit `d` in `new_B_k`.
    # 3. If no conflict, add k to S, append `new_B_k` to B.
    #    Also, we must update `B[i]` for all i in S to include the new distance `DIST[p, k]`.
    #    `B[i] |= (1 << DIST[p, k])`
    
    # This is O(|S|) per addition check, with very small constants (bitwise ops on Python ints are fast).
    # This is MUCH faster than O(|S|^2) or numpy overhead for small |S|.
    # For |S| ~ 20-50, this is extremely fast.
    
    # Let's implement this Bitset-based Greedy + SA.
    
    def try_add(k, S, B):
        """
        Tries to add point k to set S (list of indices) with bitsets B (list of ints).
        Returns True if added, False otherwise.
        Modifies S and B in place if added.
        """
        m = len(S)
        if m == 0:
            S.append(k)
            B.append(0)
            return True
        
        new_B_k = 0
        distances = DIST[k, np.array(S, dtype=np.int32)]
        
        for i in range(m):
            d = distances[i]
            # Check if distance d already exists from S[i] to any other point in S
            if (B[i] >> d) & 1:
                return False
            # Check if distance d already exists from k to any other point in S
            if (new_B_k >> d) & 1:
                return False
            new_B_k |= (1 << d)
            
        # If we get here, no conflicts.
        # Update existing B's to include distance to k
        for i in range(m):
            d = distances[i]
            B[i] |= (1 << d)
            
        S.append(k)
        B.append(new_B_k)
        return True

    def remove(k, S, B):
        """
        Removes point k from set S and updates B.
        This is tricky because we need to remove the bit corresponding to distance to k from all other B's.
        But we don't store which bit corresponds to which neighbor.
        So we must rebuild B for all remaining points? O(|S|^2).
        Or, we can just mark k as removed and ignore it? No, that breaks the logic.
        
        For SA, we need swaps.
        Swap = Remove + Add.
        If Remove is O(|S|^2), it's slow.
        
        Alternative: Don't support removal directly in the bitset structure.
        Instead, use the bitset structure for **Greedy Construction** only.
        Then use a **different** method for local search?
        
        Actually, if we just do Greedy with Random Restarts, the bitset makes it very fast.
        We can do thousands of restarts.
        
        Let's try:
        1. Greedy with Bitset (Fast).
        2. "Perturbation": Take the best set found so far, remove a random subset of points (rebuild B from scratch for the remaining points), then Greedy again.
        
        Rebuilding B from scratch for a set of size m is O(m^2).
        If m ~ 30, m^2 = 900 ops. Very fast.
        """
        pass

    # Greedy initialization with Bitset
    def greedy_init_bitset(indices_to_try):
        S = []
        B = []
        for k in indices_to_try:
            try_add(k, S, B)
        return S, B

    # Rebuild B from a list of indices S
    def rebuild_B(S):
        B = [0] * len(S)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = DIST[S[i], S[j]]
                B[i] |= (1 << d)
                B[j] |= (1 << d)
        return B

    # Main Loop
    while time.time() < deadline:
        # 1. Random Greedy
        indices = list(range(GRID_SIZE))
        random.shuffle(indices)
        S, B = greedy_init_bitset(indices)
        
        # 2. Local Search / Perturbation
        # Try to improve by removing a random point and re-greedying?
        # Or just try to add more points in a different order?
        
        # Let's try a "Swap" heuristic using the bitset check.
        # Since removal is hard, let's do:
        # Pick a random point p in S.
        # Remove p from S (conceptually).
        # Rebuild B for S \ {p}.
        # Try to add p back? No.
        # Try to add a random set of new points?
        
        # Actually, the most effective move for packing is often **Removing a random point** 
        # and then **Greedy adding** from the remaining candidates.
        
        # Let's do "Iterated Local Search" style:
        # 1. Greedy -> S
        # 2. Perturb: Remove K random points from S.
        # 3. Re-greedy on the remaining space (excluding the removed points? No, including them as candidates again).
        #    Actually, just re-greedy on ALL points, but start with the remaining S.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Perturbation
        if len(S) > 0:
            # Remove 1-3 random points
            num_remove = random.randint(1, 3)
            if num_remove > len(S):
                num_remove = len(S)
            
            remove_indices = random.sample(range(len(S)), num_remove)
            # We need to remove these from S and B.
            # Easiest: Create a new list of remaining points, rebuild B.
            
            remaining_indices = [i for i in range(len(S)) if i not in remove_indices]
            new_S = [S[i] for i in remaining_indices]
            new_B = rebuild_B(new_S)
            
            # Now, try to add more points.
            # We have a "forbidden" set of points that were just removed? No, they are candidates again.
            # We just continue greedy from this state.
            
            # To avoid re-adding the same points immediately in a bad way, 
            # we can shuffle the candidates.
            
            # Candidates are all points NOT in new_S.
            # We don't have a fast "not in new_S" check for all 4096 points.
            # But we can just shuffle ALL points and try to add them.
            # The `try_add` will reject those already in new_S (wait, it doesn't know they are in new_S).
            # We need to check `if k in new_S`.
            # `new_S` is a list. `k in new_S` is O(|S|).
            # We can use a set for `new_S` for O(1) check.
            
            S_set = set(new_S)
            
            candidates = list(range(GRID_SIZE))
            random.shuffle(candidates)
            
            for k in candidates:
                if k in S_set:
                    continue
                try_add(k, new_S, new_B)
                # Note: try_add modifies new_S and new_B.
                # We don't need to update S_set because we are iterating through a fixed list of candidates
                # and `try_add` will fail if k is already added (distance 0? No, distance to self is 0, but we don't add self).
                # Wait, `try_add` checks distances to *existing* points.
                # If k is already in new_S, `DIST[k, k]` is 0.
                # The loop in `try_add` iterates over `S` (which is `new_S`).
                # If k is in new_S, then for i where S[i]==k, d=0.
                # `B[i]` has bit 0? No, distances are > 0 for distinct points.
                # So `try_add` might try to add a point that is already in the set.
                # We must ensure we don't try to add points already in the set.
                # The `if k in S_set` check above handles this for the initial state.
                # But as we add points, `S_set` becomes stale.
                # However, `try_add` for a point k already in S:
                # It will check distances to all points in S.
                # One of them is k itself. d=0.
                # `B[i]` for i where S[i]==k does not have bit 0 set (distances to distinct points).
                # `new_B_k` will get bit 0 set.
                # Next point j in S (j!=k). d = DIST[k,j].
                # If `B[i]` has bit d set? Yes, because j is in S and connected to k.
                # So `try_add` will return False for a point already in S.
                # So we don't strictly need the `if k in S_set` check for correctness, 
                # but it saves time (skipping the loop).
                # Since `S_set` is static during the loop, it's a good filter.
                
            if len(new_S) > best_size:
                best_size = len(new_S)
                best_set = new_S[:]

    return [ALL_POINTS[i] for i in best_set]

# Run once to test syntax/logic if needed, but construct is the entry point.
