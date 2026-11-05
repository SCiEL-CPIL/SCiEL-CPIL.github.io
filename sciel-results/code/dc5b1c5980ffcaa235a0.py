import time
import random
import math

def construct():
    N = 512
    # Bit i represents element i+1
    
    # Precompute conflict masks
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            mask_bce = b_bit | c_bit | e_bit
            conflicts[a - 1].append(mask_bce)
            
            mask_ace = a_bit | c_bit | e_bit
            conflicts[b - 1].append(mask_ace)
            
            mask_abe = a_bit | b_bit | e_bit
            conflicts[c - 1].append(mask_abe)
            
            mask_abc = a_bit | a_bit | c_bit # Wait, mask_abc should be a, b, c
            mask_abc = a_bit | b_bit | c_bit
            conflicts[e - 1].append(mask_abc)

    # Precompute degrees for heuristic ordering
    degrees = [len(conflicts[i]) for i in range(N)]
    # Sort indices by degree (ascending) for initial greedy
    order_by_degree = sorted(range(N), key=lambda i: degrees[i])

    def can_add(x_idx, S):
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def greedy_fill(S, order):
        for x_idx in order:
            if not (S & (1 << x_idx)):
                if can_add(x_idx, S):
                    S |= (1 << x_idx)
        return S

    def count_bits(S):
        return bin(S).count('1')

    start_time = time.time()
    deadline = start_time + 55.0

    best_S = 0
    best_size = 0
    
    # Initial Greedy Constructions
    # 1. Degree sorted
    S_try = greedy_fill(0, order_by_degree)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try
        current_S = S_try
        current_size = size_try

    # 2. Random shuffles
    base_order = list(range(N))
    for _ in range(50):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            current_S = S_try
            current_size = size_try

    # Local Search
    iterations = 0
    stagnation = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # 1. Try to add any element (Intensification)
        # Scan all elements, if any can be added, add it and restart scan?
        # Or just do one pass? One pass is faster.
        # Let's do a pass.
        added_any = False
        # Use random order for this pass to avoid bias
        order = list(range(N))
        random.shuffle(order)
        for x_idx in order:
            if not (current_S & (1 << x_idx)):
                if can_add(x_idx, current_S):
                    current_S |= (1 << x_idx)
                    current_size += 1
                    added_any = True
        
        if added_any:
            if current_size > best_size:
                best_size = current_size
                best_S = current_S
            # If we added something, we might be able to add more.
            # Continue loop to try again.
            continue

        # 2. Stuck: No elements can be added.
        # Perform Perturbation
        # Increase kick size with stagnation
        kick_size = min(10, 1 + stagnation // 3)
        
        elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
        
        if len(elements_in_set) < kick_size:
            kick_size = len(elements_in_set)
        
        if kick_size > 0:
            removed = random.sample(elements_in_set, kick_size)
            S_temp = current_S
            for idx in removed:
                S_temp &= ~(1 << idx)
            
            # Greedy fill to repair
            order = list(range(N))
            random.shuffle(order)
            S_filled = greedy_fill(S_temp, order)
            size_filled = count_bits(S_filled)
            
            if size_filled > best_size:
                best_size = size_filled
                best_S = S_filled
                stagnation = 0
            else:
                stagnation += 1
            
            current_S = S_filled
            current_size = size_filled
        else:
            # Set is empty or too small to kick, just restart
            break

    # Final check
    if count_bits(current_S) > best_size:
        best_S = current_S
        best_size = count_bits(current_S)

    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
