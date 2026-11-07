import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin under 90s

    N = 256

    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_index = {p: i for i, p in enumerate(primes)}

    factor_masks = [0] * (N + 1)
    factor_masks[1] = 0
    for i in range(2, N + 1):
        x = i
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        factor_masks[i] = mask

    weights = [i + 1 for i in range(N)]
    
    # compat_mask[i] is bitmask of j such that gcd(i+1, j+1) == 1
    compat_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j and math.gcd(i + 1, j + 1) == 1:
                compat_mask[i] |= (1 << j)

    best_weight = 0
    best_set = []

    def greedy_init():
        R = (1 << N) - 1
        sol = []
        w = 0
        while R:
            best_v = -1
            best_w = -1
            m = R
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                m ^= lsb
            if best_v == -1:
                break
            sol.append(best_v)
            w += weights[best_v]
            R &= compat_mask[best_v]
        return w, sol

    init_w, init_sol = greedy_init()
    best_weight = init_w
    best_set = init_sol[:]

    def get_bound(R_mask):
        # Extract vertices
        verts = []
        m = R_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if not verts:
            return 0

        # Sort by weight descending
        verts.sort(key=lambda x: weights[x], reverse=True)
        
        # Greedy coloring of Compat Graph
        # color[v] = c
        # We want to assign colors such that if u, v are compatible (adjacent in Compat), they have different colors.
        # This is standard graph coloring on the Compat Graph.
        
        # To speed up, we use a list of colors assigned so far.
        # For each new vertex v, we check its neighbors in the colored set.
        # Neighbors are those u in colored_set such that (1 << u) & compat_mask[v] != 0.
        
        # We can maintain a bitmask of colored vertices for each color?
        # No, we just need to know if a color is used by a neighbor.
        
        # Let's use a dictionary: color -> bitmask of vertices with that color.
        # Then for v, we check if any neighbor has color c.
        # neighbor_mask = compat_mask[v] & colored_mask
        # If neighbor_mask & color_mask[c] != 0, then color c is used.
        
        color_masks = [] # list of bitmasks
        # We process verts in order.
        
        for v in verts:
            # Check existing colors
            assigned = False
            for c_idx, cm in enumerate(color_masks):
                # Check if v has any neighbor in cm
                # Neighbors of v in the colored set are (compat_mask[v] & colored_so_far_mask)
                # But we don't have a global colored_so_far_mask easily accessible in this loop without updating it.
                # Actually, we can just check if (compat_mask[v] & cm) != 0.
                if (compat_mask[v] & cm) == 0:
                    # No conflict, assign this color
                    color_masks[c_idx] = cm | (1 << v)
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(1 << v)
                
        # Bound is sum of max weights in each color class
        # We need to find max weight for each color class.
        # Since we assigned vertices in descending weight order, the FIRST vertex assigned to a color class is the max weight in that class.
        # So we just need to store the weight of the first vertex assigned to each color.
        
        # Let's re-do the coloring to track max weights.
        
    # Re-implement get_bound with weight tracking
    def get_bound_optimized(R_mask):
        verts = []
        m = R_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if not verts:
            return 0

        verts.sort(key=lambda x: weights[x], reverse=True)
        
        # color_max_weight[c] = max weight of vertex in color c
        # color_mask[c] = bitmask of vertices in color c
        color_max_weight = []
        color_mask = []
        
        for v in verts:
            assigned = False
            w_v = weights[v]
            for c_idx in range(len(color_mask)):
                # Check if v can take color c_idx
                # v can take color c_idx if it is NOT adjacent to any vertex in color_mask[c_idx] in the Compat Graph.
                # i.e., (compat_mask[v] & color_mask[c_idx]) == 0
                if (compat_mask[v] & color_mask[c_idx]) == 0:
                    color_mask[c_idx] |= (1 << v)
                    # Since we process in descending weight, the first one is the max.
                    # So we don't need to update max weight if we only add if it's the first?
                    # No, we add to the existing class. The max weight is already set to the first vertex.
                    assigned = True
                    break
            
            if not assigned:
                color_mask.append(1 << v)
                color_max_weight.append(w_v)
                
        return sum(color_max_weight)

    def solve(R, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline - 2:
            return
            
        if R == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = current_set[:]
            return
            
        # Bound
        # Weak bound: sum of weights
        sum_w = 0
        m = R
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            sum_w += weights[idx]
            m ^= lsb
            
        if current_weight + sum_w <= best_weight:
            return
            
        # Strong bound
        # Only use if weak bound is promising?
        # Or always use if R is small?
        # Let's use it if sum_w is not too large to save time on trivial prunes?
        # Actually, coloring bound is O(V^2).
        # If V is large, it's expensive.
        # But it prunes more.
        
        bound = get_bound_optimized(R)
        if current_weight + bound <= best_weight:
            return
            
        # Branch
        # Pick vertex v
        # Heuristic: Pick vertex with highest weight in R
        v = -1
        max_w = -1
        m = R
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        new_R = R & compat_mask[v]
        solve(new_R, current_weight + weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        new_R2 = R & ~(1 << v)
        solve(new_R2, current_weight, current_set)

    full_mask = (1 << N) - 1
    solve(full_mask, 0, [])
    
    result = [v + 1 for v in best_set]
    
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(result):
        result = [v + 1 for v in init_sol]
        
    return result

