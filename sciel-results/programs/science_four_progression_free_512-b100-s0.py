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
            
            mask_abc = a_bit | b_bit | c_bit
            conflicts[e - 1].append(mask_abc)

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
    current_S = 0
    current_size = 0

    # Initial Greedy Constructions
    base_order = list(range(N))
    
    S_try = greedy_fill(0, base_order)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try
        current_S = S_try
        current_size = size_try

    for _ in range(30):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            current_S = S_try
            current_size = size_try

    # Local Search
    # We will use a combination of "Remove and Fill" and "Swap" moves.
    # Also, we will use a "Periodic Fill" strategy: after every few moves, 
    # we run a full greedy fill to ensure we are at a local optimum.
    
    iterations = 0
    stagnation = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # 1. Periodic Full Fill (to escape shallow local optima)
        # This is cheap if we just check additions.
        # But greedy_fill is O(N * K). 
        # Let's do it every 10 iterations or so.
        if iterations % 10 == 0:
            # Try to add any element that can be added, in random order
            order = list(range(N))
            random.shuffle(order)
            S_filled = greedy_fill(current_S, order)
            size_filled = count_bits(S_filled)
            if size_filled > best_size:
                best_size = size_filled
                best_S = S_filled
                stagnation = 0
            else:
                stagnation += 1
            current_S = S_filled
            current_size = size_filled
            continue

        # 2. Move Type Selection
        # If stagnation is high, increase perturbation strength
        perturbation_strength = min(5, 1 + stagnation // 5)
        
        if random.random() < 0.7:
            # Remove and Fill
            k = random.randint(1, perturbation_strength)
            
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            
            if len(elements_in_set) < k:
                k = len(elements_in_set)
                if k == 0:
                    continue
            
            if k > 0:
                removed = random.sample(elements_in_set, k)
                S_temp = current_S
                for idx in removed:
                    S_temp &= ~(1 << idx)
                
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
            # Swap Move
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            if not elements_in_set:
                continue
            
            u = random.choice(elements_in_set)
            S_temp = current_S & ~(1 << u)
            
            # Try to find a v that can be added
            # We can bias towards elements that were recently removed or just random
            # Let's just try random v's
            found_swap = False
            for _ in range(20):
                v = random.randint(0, N - 1)
                if current_S & (1 << v):
                    continue
                
                if can_add(v, S_temp):
                    S_new = S_temp | (1 << v)
                    current_S = S_new
                    # Size is same, but structure changed
                    # Check if we can now fill more
                    order = list(range(N))
                    random.shuffle(order)
                    S_filled = greedy_fill(current_S, order)
                    size_filled = count_bits(S_filled)
                    
                    if size_filled > best_size:
                        best_size = size_filled
                        best_S = S_filled
                        stagnation = 0
                    else:
                        stagnation += 1
                    
                    current_S = S_filled
                    current_size = size_filled
                    found_swap = True
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
