import itertools
import random
import time

N = 16
N2 = N * N

# Precompute point coordinates
points = list(itertools.product(range(N), repeat=2))

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute iso_with[i][j] as bitset of k such that i, j, k is isosceles
iso_with = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        mask = 0
        for k in range(N2):
            if k == i or k == j:
                continue
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                mask |= (1 << k)
        iso_with[i][j] = mask

def get_blocked_mask(S_indices):
    """
    Returns a bitmask of all points that form an isosceles triangle 
    with at least one pair of points in S_indices.
    """
    mask = 0
    n = len(S_indices)
    for i in range(n):
        a = S_indices[i]
        for j in range(i + 1, n):
            b = S_indices[j]
            mask |= iso_with[a][b]
    return mask

def is_maximal(S_indices):
    """
    Checks if S_indices is a maximal no-isosceles set.
    """
    if not S_indices:
        return False
        
    S_mask = 0
    for p in S_indices:
        S_mask |= (1 << p)
        
    all_points_mask = (1 << N2) - 1
    not_S_mask = all_points_mask ^ S_mask
    
    # Compute B(S): Union of all points that form an isosceles triangle with any pair in S
    B_mask = get_blocked_mask(S_indices)
    
    # S is maximal if every point not in S is in B(S)
    return (B_mask & not_S_mask) == not_S_mask

def greedy_construct(order):
    chosen = []
    S_mask = 0
    for p in order:
        # Check if p can be added
        # p is valid if for all a in chosen, iso_with[p][a] & S_mask == 0
        # Wait, if iso_with[p][a] & S_mask != 0, it means there exists b in S such that {p, a, b} is isosceles.
        valid = True
        for a in chosen:
            if (iso_with[p][a] & S_mask):
                valid = False
                break
        if valid:
            chosen.append(p)
            S_mask |= (1 << p)
    return chosen

def shrink(S):
    S = list(S)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(S)))
        random.shuffle(indices)
        for idx in indices:
            p = S[idx]
            candidate = S[:idx] + S[idx + 1:]
            if is_maximal(candidate):
                S = candidate
                improved = True
                break
    return S

def local_search(S):
    """
    Performs local search with removals and swaps to minimize size of maximal set.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Try Removals
        indices = list(range(len(S)))
        random.shuffle(indices)
        for idx in indices:
            p = S[idx]
            candidate = S[:idx] + S[idx + 1:]
            if is_maximal(candidate):
                S = candidate
                improved = True
                break
        
        if improved:
            continue # Restart loop with new S

        # Phase 2: Try Swaps
        # If we can't remove, try to swap to a different configuration of same size
        # This might allow further removals later.
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
            
        for idx in indices:
            p_out = S[idx]
            S_temp = S[:idx] + S[idx + 1:]
            
            # Calculate blocked mask for S_temp
            B_temp = get_blocked_mask(S_temp)
            
            # Candidates for p_in:
            # 1. Not in S (S_mask)
            # 2. Not in B_temp (to ensure validity)
            # 3. Not p_out (though p_out is in S_mask, so it's excluded by 1)
            
            all_mask = (1 << N2) - 1
            # Points not in S
            not_S_mask = all_mask ^ S_mask
            # Points not in B_temp
            not_B_temp_mask = all_mask ^ B_temp
            
            candidates_mask = not_S_mask & not_B_temp_mask
            
            # Iterate through candidates
            # We want to find ANY candidate that results in a maximal set.
            # To speed up, we can just iterate and check.
            c = candidates_mask
            while c:
                # Extract lowest set bit
                lsb = c & (-c)
                p_in = lsb.bit_length() - 1
                c ^= lsb
                
                # Form new set
                S_new = S_temp + [p_in]
                
                # Check maximality
                if is_maximal(S_new):
                    S = S_new
                    improved = True
                    break
            
            if improved:
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = float('inf')
    best_set = []
    
    # Precompute degree for heuristic
    degrees = [0] * N2
    for i in range(N2):
        for j in range(N2):
            if i != j and iso_with[i][j]:
                degrees[i] += 1
                
    order_high = sorted(range(N2), key=lambda x: -degrees[x])
    order_low = sorted(range(N2), key=lambda x: degrees[x])
    
    while time.time() < deadline - 10:
        if random.random() < 0.5:
            base_order = order_high
        else:
            base_order = order_low
            
        order = list(base_order)
        random.shuffle(order)
        
        S = greedy_construct(order)
        
        # Shrink first
        S_shrunk = shrink(S)
        
        # Then Local Search
        S_ls = local_search(S_shrunk)
        
        if len(S_ls) < best_size:
            best_size = len(S_ls)
            best_set = S_ls[:]
            
    return [points[i] for i in best_set]
