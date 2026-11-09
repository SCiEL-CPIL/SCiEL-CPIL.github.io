import time
import itertools
import sys

N = 16
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # Indexing: point (x, y) -> x * N + y
    # Precompute isosceles conflicts
    # conflict[i][j] is a bitmask of points k such that {i, j, k} is isosceles
    # Note: i and j themselves are not included in the mask, but we don't need to add them to blocked
    # because we track chosen separately.
    
    points = list(itertools.product(range(N), repeat=2))
    idx_to_pt = {i: p for i, p in enumerate(points)}
    pt_to_idx = {p: i for i, p in enumerate(points)}
    
    # We need to know for any pair (i, j), which k makes it isosceles.
    # d2(a, b)
    def d2(i, j):
        ax, ay = idx_to_pt[i]
        bx, by = idx_to_pt[j]
        return (ax - bx)**2 + (ay - by)**2

    # Precompute conflicts
    # conflicts[i] is a list of bitmasks? No, conflicts[i][j] is a bitmask.
    # But storing a 256x256 matrix of 256-bit integers is ~256*256*32 bytes = 2MB. Acceptable.
    # Actually, we can just compute on the fly or precompute a list of lists.
    
    # Let's precompute `iso_mask[i][j]`
    # i, j in 0..255
    # This is O(256^2 * 256) = 16M ops. Fast enough.
    
    iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        ax, ay = idx_to_pt[i]
        for j in range(i + 1, TOTAL_POINTS):
            bx, by = idx_to_pt[j]
            d_ij = d2(i, j)
            
            # Find all k such that d(i,k) == d(i,j) or d(j,k) == d(i,j) or d(i,k) == d(j,k)
            mask = 0
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                cx, cy = idx_to_pt[k]
                d_ik = (ax - cx)**2 + (ay - cy)**2
                d_jk = (bx - cx)**2 + (by - cy)**2
                
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    mask |= (1 << k)
            
            iso_mask[i][j] = mask
            iso_mask[j][i] = mask

    all_mask = (1 << TOTAL_POINTS) - 1
    
    best_set_mask = None
    best_size = TOTAL_POINTS + 1
    
    # Backtracking search
    # State: chosen_mask, blocked_mask
    # available = all_mask & ~chosen_mask & ~blocked_mask
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(chosen_mask, blocked_mask, depth):
        nonlocal best_set_mask, best_size, deadline
        
        if time.time() > deadline:
            return
        
        # Check if we have a maximal set
        # A set is maximal if no point can be added.
        # Points that can be added are those not in chosen and not in blocked.
        available = all_mask & ~chosen_mask & ~blocked_mask
        
        if available == 0:
            # We found a maximal set
            size = bin(chosen_mask).count('1')
            if size < best_size:
                best_size = size
                best_set_mask = chosen_mask
            return
            
        # Pruning: If current depth + lower_bound > best_size, return
        # Lower bound: at least 1 more point is needed if available != 0
        # A stronger bound: each new point blocks some others.
        # Simple bound: depth + 1 > best_size? No, that's weak.
        
        # Heuristic: Pick a point from available to add.
        # Strategy: Pick the point that blocks the most other available points.
        # This is a greedy choice within the backtracking to guide the search.
        
        # To avoid trying all 256 points at every step (which might be too slow),
        # we can limit the branching factor or use a more aggressive heuristic.
        # Let's iterate through available points, but maybe sort them by "impact".
        
        # Get list of available indices
        # available is a bitmask.
        avail_list = []
        tmp = available
        while tmp:
            # Get lowest set bit
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            tmp ^= lsb
        
        if not avail_list:
            return

        # Sort by impact: how many new points does adding this point block?
        # Impact of adding p:
        # new_blocked = 0
        # for each c in chosen:
        #     new_blocked |= iso_mask[p][c]
        # impact = bin(new_blocked & available).count('1')
        
        scored_points = []
        for p in avail_list:
            new_blocked = 0
            # Iterate chosen points
            tmp_c = chosen_mask
            while tmp_c:
                lsb_c = tmp_c & -tmp_c
                c_idx = lsb_c.bit_length() - 1
                new_blocked |= iso_mask[p][c_idx]
                tmp_c ^= lsb_c
            
            # Count how many currently available points are blocked by this choice
            # Note: p itself is in available, so it will be removed from available anyway.
            # We care about blocking OTHER available points.
            impact = bin(new_blocked & available).count('1')
            scored_points.append((impact, p))
            
        # Sort by impact descending (greedy heuristic)
        scored_points.sort(key=lambda x: x[0], reverse=True)
        
        # To prevent explosion, we might limit the number of branches explored.
        # However, for N=16, the search space is small enough if we prune well.
        # Let's try all branches but rely on the time limit.
        # If it's too slow, we can limit to top K branches.
        
        # Optimization: If we find a solution, we can prune branches that can't beat it.
        
        for _, p in scored_points:
            if time.time() > deadline:
                return
            
            # Add p
            new_chosen = chosen_mask | (1 << p)
            
            # Update blocked
            new_blocked = blocked_mask
            tmp_c = chosen_mask
            while tmp_c:
                lsb_c = tmp_c & -tmp_c
                c_idx = lsb_c.bit_length() - 1
                new_blocked |= iso_mask[p][c_idx]
                tmp_c ^= lsb_c
            
            # Recurse
            dfs(new_chosen, new_blocked, depth + 1)

    # Initial call
    dfs(0, 0, 0)
    
    # If no solution found (shouldn't happen, greedy always finds one), fallback
    if best_set_mask is None:
        # Fallback to simple greedy
        chosen_mask = 0
        blocked_mask = 0
        available = all_mask
        while available:
            # Pick first available
            lsb = available & -available
            p = lsb.bit_length() - 1
            chosen_mask |= (1 << p)
            
            # Update blocked
            tmp_c = chosen_mask & ~(1 << p) # chosen without p
            while tmp_c:
                lsb_c = tmp_c & -tmp_c
                c_idx = lsb_c.bit_length() - 1
                blocked_mask |= iso_mask[p][c_idx]
                tmp_c ^= lsb_c
            
            available = all_mask & ~chosen_mask & ~blocked_mask
        
        best_set_mask = chosen_mask
    
    # Convert mask to list of tuples
    result = []
    tmp = best_set_mask
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        result.append(idx_to_pt[idx])
        tmp ^= lsb
        
    return result
