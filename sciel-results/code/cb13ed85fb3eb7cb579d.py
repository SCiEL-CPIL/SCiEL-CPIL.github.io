import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    n = 256
    nums = list(range(2, 257))
    m = len(nums)  # 255
    
    # Precompute compatibility bitmask
    # compat[i] is an integer bitmask where bit j is 1 if lcm(nums[i], nums[j]) > 256
    compat = [0] * m
    for i in range(m):
        mask = 0
        ni = nums[i]
        for j in range(m):
            if i == j:
                continue
            nj = nums[j]
            if lcm(ni, nj) > 256:
                mask |= (1 << j)
        compat[i] = mask
        
    weights = [1.0 / nums[i] for i in range(m)]
    
    best_weight = 0.0
    best_clique_mask = 0
    
    # Bron-Kerbosch with bitmasks
    # R, P, X are bitmasks
    # r_weight is the sum of weights in R
    
    def bron_kerbosch(R, P, X, r_weight):
        nonlocal best_weight, best_clique_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if r_weight > best_weight:
                best_weight = r_weight
                best_clique_mask = R
            return
            
        if P == 0:
            return
            
        # Bound: sum of weights of vertices in P
        # This is a loose bound. A tighter bound would use coloring.
        # For now, let's just check if adding all of P could beat best.
        # We can precompute prefix sums or just iterate? 
        # Iterating over bits in P is fast.
        bound = 0.0
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            p_temp ^= lsb
            
        if r_weight + bound <= best_weight:
            return
            
        # Pivot selection
        # Choose u in P | X that maximizes popcount(P & compat[u])
        # This is expensive to do exactly. 
        # Heuristic: pick u from P that has fewest neighbors in P? 
        # Or just pick first one? 
        # For now, let's skip pivot or use a simple one to save time on pivot selection logic.
        # Actually, pivot is crucial for performance.
        
        # Simple pivot: pick u from P with max degree in P?
        # Or just iterate all v in P.
        
        # Let's try iterating v in P in order of decreasing weight.
        # This helps find good solutions early.
        
        # Get list of indices in P sorted by weight descending
        p_indices = []
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            idx = lsb.bit_length() - 1
            p_indices.append(idx)
            p_temp ^= lsb
            
        p_indices.sort(key=lambda i: weights[i], reverse=True)
        
        for v in p_indices:
            if time.time() > deadline:
                return
            
            # New P = P \ {v} & compat[v]
            new_P = P & compat[v] & ~(1 << v)
            # New X = X & compat[v]
            new_X = X & compat[v]
            
            R_new = R | (1 << v)
            
            bron_kerbosch(R_new, new_P, new_X, r_weight + weights[v])
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            if P == 0 and X == 0:
                break

    initial_P = (1 << m) - 1
    initial_X = 0
    initial_R = 0
    
    bron_kerbosch(initial_R, initial_P, initial_X, 0.0)
    
    # Convert best_clique_mask to list of numbers
    result = []
    mask = best_clique_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(nums[idx])
        mask ^= lsb
        
    if not result:
        result = [2]
        
    return result
