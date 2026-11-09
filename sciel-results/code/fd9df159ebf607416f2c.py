"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute masks for 4-AP conflicts and 3-AP gains
    # conflict_masks[x] contains bitmasks of the other 3 elements for each 4-AP x is in
    # gain_masks[x] contains bitmasks of the other 2 elements for each 3-AP x is in
    
    conflict_masks = [[] for _ in range(N + 1)]
    gain_masks = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            # 4-AP: a, b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            mask_b = (1 << a) | (1 << c) | (1 << e)
            mask_c = (1 << a) | (1 << b) | (1 << e)
            mask_e = (1 << a) | (1 << b) | (1 << c)
            
            conflict_masks[a].append(mask_a)
            conflict_masks[b].append(mask_b)
            conflict_masks[c].append(mask_c)
            conflict_masks[e].append(mask_e)
            
    # Generate all 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            
            # 3-AP: a, b, c
            mask_a = (1 << b) | (1 << c)
            mask_b = (1 << a) | (1 << c)
            mask_c = (1 << a) | (1 << b)
            
            gain_masks[a].append(mask_a)
            gain_masks[b].append(mask_b)
            gain_masks[c].append(mask_c)

    def is_valid_add(x, M):
        # M is bitmask of current set A (without x)
        for mask in conflict_masks[x]:
            if (M & mask) == mask:
                return False
        return True

    def count_3ap_gain(x, M):
        # Count 3-APs formed by adding x to M
        gain = 0
        for mask in gain_masks[x]:
            if (M & mask) == mask:
                gain += 1
        return gain

    def count_total_3ap(M):
        # Calculate total 3-APs in M
        # Sum gain(x, M) for x in M, divide by 3
        total = 0
        m = M
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            total += count_3ap_gain(x, M)
            m ^= lsb
        return total // 3

    def get_elements(M):
        """Extract elements from bitmask."""
        elems = []
        m = M
        while m:
            lsb = m & -m
            elems.append(lsb.bit_length() - 1)
            m ^= lsb
        return elems

    def local_search(M, current_score):
        """Perform local search with additions and swaps."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try single additions
            for x in range(1, N + 1):
                if not (M & (1 << x)):
                    if is_valid_add(x, M):
                        gain = count_3ap_gain(x, M)
                        if gain > 0:
                            new_M = M | (1 << x)
                            new_score = current_score + gain
                            if new_score > current_score:
                                M = new_M
                                current_score = new_score
                                improved = True
                                break
            
            # 2. Try swaps: remove a, add b
            A = get_elements(M)
            B = [x for x in range(1, N + 1) if not (M & (1 << x))]
            
            best_delta = 0
            best_a = -1
            best_b = -1
            
            for a in A:
                if time.time() > deadline - 5:
                    return M, current_score
                M_temp = M & ~(1 << a)
                loss_a = count_3ap_gain(a, M) # Number of 3-APs in M containing a
                
                for b in B:
                    if not is_valid_add(b, M_temp):
                        continue
                    gain_b = count_3ap_gain(b, M_temp)
                    delta = gain_b - loss_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                M = M & ~(1 << best_a) | (1 << best_b)
                current_score += best_delta
                improved = True
                
        return M, current_score

    def greedy_construct():
        """Greedy construction to get a starting point."""
        M = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        while remaining and time.time() < deadline - 10:
            best_x = -1
            best_gain = -1
            for x in remaining:
                if is_valid_add(x, M):
                    g = count_3ap_gain(x, M)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            if best_x == -1:
                break
            M |= (1 << best_x)
            remaining.remove(best_x)
        
        return M

    best_M = 0
    best_score = 0
    
    # Initial Greedy Construction with multiple restarts
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        M = greedy_construct()
        M, score = local_search(M, count_total_3ap(M))
        if score > best_score:
            best_score = score
            best_M = M

    # Iterated Local Search with simulated annealing-inspired perturbation
    M = best_M
    current_score = best_score
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline - 2 and T > T_min:
        # Local Search
        M, current_score = local_search(M, current_score)
        
        if current_score > best_score:
            best_score = current_score
            best_M = M
            
        # Perturbation: Randomly remove k elements
        k = random.randint(3, 8)
        A = get_elements(M)
        
        if len(A) > k:
            to_remove = random.sample(A, k)
            for x in to_remove:
                M &= ~(1 << x)
            current_score = count_total_3ap(M)
        else:
            # If set is too small, just restart with greedy
            M = greedy_construct()
            current_score = count_total_3ap(M)
        
        T *= cooling_rate
    
    # Final local search on best solution
    best_M, best_score = local_search(best_M, best_score)
    
    # Convert bitmask to sorted list
    result = get_elements(best_M)
    return sorted(result)
# EVOLVE-BLOCK-END
