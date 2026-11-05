import time
import random
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

# Precompute isosceles neighbors
# iso_map[j][u] is a set of v such that (j, u, v) forms an isosceles triangle
# Note: The triangle is isosceles if any two sides are equal.
# We store this for efficient conflict checking.
# To save memory and time, we only store if v != j and v != u.
iso_map = [dict() for _ in range(TOTAL)]

for j in range(TOTAL):
    for u in range(TOTAL):
        if j == u:
            continue
        d_ju = dist2[j, u]
        # Find v such that:
        # 1. d(j,v) == d(j,u)  (Apex at j)
        # 2. d(u,v) == d(j,u)  (Apex at u)
        # 3. d(j,v) == d(u,v)  (Apex at v)
        
        # We can iterate v, but that's O(N^3).
        # Better:
        # For case 1: v is on circle centered j, radius d_ju.
        # For case 2: v is on circle centered u, radius d_ju.
        # For case 3: v is on perpendicular bisector of ju.
        
        # Given N=64, O(N^3) is 4096^3 = 68 billion. Too slow.
        # We must use the geometric properties or precompute circles.
        
        # Let's precompute circles:
        # circle[center][dist2] = list of points at that squared distance
        circles = [dict() for _ in range(TOTAL)]
        for c in range(TOTAL):
            for p in range(TOTAL):
                if c == p:
                    continue
                d = dist2[c, p]
                if d not in circles[c]:
                    circles[c][d] = []
                circles[c][d].append(p)

        # Now for fixed j, u:
        # Case 1: d(j,v) == d(j,u) => v in circles[j][d_ju]
        # Case 2: d(u,v) == d(j,u) => v in circles[u][d_ju]
        # Case 3: d(j,v) == d(u,v) => v is equidistant from j and u.
        #         This means 2*d(j,v)^2 = d(j,u)^2 + ... no.
        #         d(j,v)^2 = d(u,v)^2
        #         (xj-xv)^2 + (yj-yv)^2 = (xu-xv)^2 + (yu-yv)^2
        #         This is a linear equation in xv, yv. It's the perpendicular bisector.
        #         We can just iterate all v and check d(j,v) == d(u,v).
        #         But we can optimize:
        #         The set of such v is small.
        
        # Let's just iterate v for Case 3, and use circles for 1 and 2.
        # Actually, iterating v for all j,u is O(N^3).
        # 4096 * 4096 * 4096 is too much.
        
        # Alternative:
        # Just iterate v for all j,u,v but break early?
        # Or, precompute for each pair (j,u) the list of v.
        # There are ~16 million pairs.
        # For each pair, finding v takes O(N) if we scan. 16M * 4096 = 68B. Too slow.
        
        # We need a faster way to find v for Case 3.
        # d(j,v) == d(u,v)
        # Let j=(x1,y1), u=(x2,y2), v=(x,y).
        # (x-x1)^2 + (y-y1)^2 = (x-x2)^2 + (y-y2)^2
        # x^2 - 2x1x + x1^2 + y^2 - 2y1y + y1^2 = x^2 - 2x2x + x2^2 + y^2 - 2y2y + y2^2
        # 2(x2-x1)x + 2(y2-y1)y = x2^2 + y2^2 - x1^2 - y1^2
        # This is a line equation.
        # We can precompute for each line the points on it?
        # Or just check the condition for each v?
        
        # Let's optimize the precomputation loop.
        # We only need iso_map for the local search.
        # Maybe we don't need to precompute ALL pairs.
        # We can compute on the fly?
        # In local search, we check a candidate j against current S.
        # S is small (~15 points).
        # So we only need to check v for the u in S.
        # So we don't need the full iso_map.
        # We can compute the conflicts for a specific j and specific S on the fly.
        
        # Conflict count for j given S:
        # count = 0
        # for u in S:
        #   for v in S:
        #     if u == v: continue
        #     if d(j,u) == d(j,v) or d(j,u) == d(u,v) or d(u,v) == d(j,v):
        #        count += 1 # Careful with double counting?
        #        # Each triangle is counted 3 times (once for each apex)?
        #        # We just need to know if count > 0.
        
        # This is O(|S|^2). |S| ~ 15. 225 checks.
        # Each check is O(1) distance lookup.
        # This is VERY fast.
        # We don't need precomputed iso_map!

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    # Initial solution using a fast greedy
    # We'll use the logic from the current program but optimized
    best_set = []
    best_size = 0
    
    # Run a few greedy restarts to get a good initial S
    for attempt in range(50):
        if time.time() > deadline:
            break
        perm = list(range(TOTAL))
        random.shuffle(perm)
        si = []
        # Use a set for fast membership
        si_set = set()
        
        for p_idx in perm:
            valid = True
            # Check against all pairs in si
            # Optimization: check distances to all in si first
            dists = []
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            # Check distinctness for isosceles with p as apex
            if len(set(dists)) != len(dists):
                continue
                
            # Check if p forms isosceles with any pair in si
            # For each pair (a, b) in si, check if d(p,a)==d(p,b) or d(p,a)==d(a,b) or d(a,b)==d(p,b)
            # We already know d(p,a) and d(p,b) are distinct (from above).
            # So we only need to check if d(p,a) == d(a,b) or d(p,b) == d(a,b).
            # i.e., if d(a,b) is in {d(p,a), d(p,b)}
            
            # Precompute d(a,b) for all pairs in si?
            # si is small.
            for i in range(len(si)):
                for j in range(i+1, len(si)):
                    a = si[i]
                    b = si[j]
                    d_ab = dist2[a, b]
                    d_pa = dists[i]
                    d_pb = dists[j]
                    
                    # Isosceles conditions:
                    # 1. d_pa == d_pb (Already checked, false)
                    # 2. d_pa == d_ab
                    # 3. d_pb == d_ab
                    if d_pa == d_ab or d_pb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                si.append(p_idx)
                si_set.add(p_idx)
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = si[:]
            best_set_copy = set(si)

    # Local Search / Hill Climbing on the best set found
    # State: best_set (list), best_set_copy (set)
    # We try to add points or swap points.
    
    current_set = best_set[:]
    current_set_set = set(best_set)
    current_size = len(current_set)
    
    # To speed up conflict checking, we can maintain a list of points.
    # Check if candidate c can be added:
    def can_add(c, S_list, S_set):
        # S_list is list of indices
        dists = []
        for q in S_list:
            dists.append(dist2[c, q])
        
        # 1. Check apex at c
        if len(set(dists)) != len(dists):
            return False
            
        # 2. Check apex at q or r for pairs (q, r) in S
        # We need to check if d(q,r) equals d(c,q) or d(c,r)
        for i in range(len(S_list)):
            qi = S_list[i]
            d_ci = dists[i]
            for j in range(i+1, len(S_list)):
                rj = S_list[j]
                d_cj = dists[j]
                d_qr = dist2[qi, rj]
                
                if d_ci == d_qr or d_cj == d_qr:
                    return False
        return True

    # Main Local Search Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Try to add a random point
        # Pick a candidate from points not in current_set
        # To avoid O(N^2) scan for candidates, we can maintain a list of unselected points
        # But N=4096, scanning is fine.
        
        # Random candidate
        c = random.randint(0, TOTAL - 1)
        if c in current_set_set:
            continue
            
        if can_add(c, current_set, current_set_set):
            current_set.append(c)
            current_set_set.add(c)
            current_size += 1
            continue
            
        # If cannot add, try a swap
        # Find a point in current_set to remove
        # Ideally one that conflicts with c.
        # We can just pick a random point from current_set
        if not current_set:
            continue
            
        # Pick a random point to remove
        # To make it smarter, we could pick one that is "far" from c or "close"?
        # For now, random.
        remove_idx = random.randint(0, len(current_set) - 1)
        r = current_set[remove_idx]
        
        # Simulate removal
        # Create a temporary list without r
        # This is O(N).
        # Since we do this many times, we can optimize.
        # But N is small (~15-20).
        
        # Check if c can be added to S \ {r}
        # We can reuse the logic.
        
        # Optimization: 
        # The conflicts preventing c were due to pairs (q, r') in S.
        # If we remove r, only pairs involving r are gone.
        # We can check if c is valid in S \ {r} by checking:
        # 1. Distances from c to S\{r} are distinct.
        # 2. No pair (q, r') in S\{r} has d(q,r') in {d(c,q), d(c,r')}.
        
        # Let's just call can_add on the modified list.
        # To avoid list copying, we can pass a flag or modify in place.
        # Given small size, list copying is fine.
        
        temp_list = current_set[:remove_idx] + current_set[remove_idx+1:]
        
        if can_add(c, temp_list, current_set_set - {r}): # Note: set difference is slow, use temp set or just check membership
             # Actually can_add uses S_list for iteration, S_set for nothing? 
             # My can_add doesn't use S_set. It uses S_list.
             # So we just need temp_list.
             pass
        # Re-evaluate can_add properly
        # My can_add implementation above only takes S_list.
        
        if can_add(c, temp_list):
            # Perform swap
            current_set[remove_idx] = c
            current_set_set.remove(r)
            current_set_set.add(c)
            # Size remains same
            continue
            
        # If swap fails, try pure removal? 
        # Pure removal reduces size, so we only do it if we are stuck and want to explore?
        # Usually we want to maximize size.
        # If we can't add or swap, we might be in a local optimum.
        # We can try removing a random point and then adding a random one (2-opt).
        
        # 2-opt: Remove r1, then try to add c.
        # We already tried swapping r with c.
        # What if we remove r, and then add a DIFFERENT point c2?
        # This is complex.
        
        # Let's stick to Add and Swap.
        # To escape local optima, we can occasionally allow a "bad" move (Simulated Annealing).
        # But for now, let's just run many iterations.

    # Convert to output format
    return [(idx // N, idx % N) for idx in current_set]

# Wait, the can_add function inside construct is redefined every time? No, it's local.
# But dist2 is global.

# One issue: can_add is called very frequently.
# We can optimize it by precomputing d(q,r) for the current set?
# Since the set changes, we can't easily precompute.
# But we can maintain a dictionary of d(q,r) for the current set.

# Let's refine the local search to maintain pair distances.

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    # Initial greedy
    best_set = []
    for attempt in range(20):
        if time.time() > deadline:
            break
        perm = list(range(TOTAL))
        random.shuffle(perm)
        si = []
        for p_idx in perm:
            valid = True
            dists = []
            for q_idx in si:
                dists.append(dist2[p_idx, q_idx])
            if len(set(dists)) != len(dists):
                continue
            for i in range(len(si)):
                for j in range(i+1, len(si)):
                    a = si[i]
                    b = si[j]
                    d_ab = dist2[a, b]
                    d_pa = dists[i]
                    d_pb = dists[j]
                    if d_pa == d_ab or d_pb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                si.append(p_idx)
        if len(si) > len(best_set):
            best_set = si[:]

    current_set = best_set[:]
    current_set_set = set(current_set)
    
    # Maintain pair distances for current_set
    # pair_dist[i][j] = dist2[current_set[i]][current_set[j]]
    # This is a matrix of size K x K.
    
    def get_pair_dist_matrix(S):
        K = len(S)
        M = [[0]*K for _ in range(K)]
        for i in range(K):
            for j in range(i+1, K):
                d = dist2[S[i]][S[j]]
                M[i][j] = d
                M[j][i] = d
        return M

    pair_dist = get_pair_dist_matrix(current_set)
    
    def can_add_fast(c, S, pair_dist, S_set):
        K = len(S)
        # Compute distances from c to all in S
        dists = [dist2[c][S[i]] for i in range(K)]
        
        # Check apex at c
        if len(set(dists)) != K:
            return False
            
        # Check apex at S[i] or S[j]
        # For each pair (i, j), check if pair_dist[i][j] is in {dists[i], dists[j]}
        for i in range(K):
            di = dists[i]
            for j in range(i+1, K):
                dj = dists[j]
                d_ij = pair_dist[i][j]
                if di == d_ij or dj == d_ij:
                    return False
        return True

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        c = random.randint(0, TOTAL - 1)
        if c in current_set_set:
            continue
            
        if can_add_fast(c, current_set, pair_dist, current_set_set):
            # Add c
            current_set.append(c)
            current_set_set.add(c)
            # Update pair_dist
            K = len(current_set) - 1
            new_row = [0] * (K + 1)
            new_col = [0] * (K + 1)
            for i in range(K):
                d = dist2[c][current_set[i]]
                new_row[i] = d
                new_col[i] = d
            # Append row and col
            for i in range(K):
                pair_dist[i].append(new_col[i])
            pair_dist.append(new_row)
            continue
            
        # Try swap
        if not current_set:
            continue
            
        r_idx = random.randint(0, len(current_set) - 1)
        r = current_set[r_idx]
        
        # Temporarily remove r
        # We need to check if c can be added to S \ {r}
        # We can construct the temp list and temp pair_dist.
        # Since K is small, this is fine.
        
        temp_set = current_set[:r_idx] + current_set[r_idx+1:]
        
        # Reconstruct pair_dist for temp_set?
        # Or just use the slow can_add?
        # Given K is small, slow can_add is O(K^2).
        # Let's use slow can_add for the swap check to avoid complexity.
        
        def can_add_slow(c, S):
            K = len(S)
            dists = [dist2[c][S[i]] for i in range(K)]
            if len(set(dists)) != K:
                return False
            for i in range(K):
                for j in range(i+1, K):
                    d_ij = dist2[S[i]][S[j]]
                    if dists[i] == d_ij or dists[j] == d_ij:
                        return False
            return True

        if can_add_slow(c, temp_set):
            # Perform swap
            current_set[r_idx] = c
            current_set_set.remove(r)
            current_set_set.add(c)
            # Rebuild pair_dist (expensive but rare)
            pair_dist = get_pair_dist_matrix(current_set)
            continue

    return [(idx // N, idx % N) for idx in current_set]
