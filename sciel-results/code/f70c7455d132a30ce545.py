import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    N = 128
    
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count("1")

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad triples bitmask B[i][j]
    # B[i][j] is a mask of k such that {i, j, k} is a bad triple.
    B = [[0] * (N + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            B[i][j] = mask
            B[j][i] = mask

    def is_valid_addition(x, S_mask):
        """
        Check if x can be added to the set represented by S_mask.
        """
        if (S_mask >> x) & 1:
            return False
            
        temp = S_mask
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            
            # Check if any z in S is in B[x][y]
            if (B[x][y] & S_mask) != 0:
                return False
                
            temp ^= lsb
            
        return True

    def greedy_build(order):
        s = 0
        for x in order:
            if is_valid_addition(x, s):
                s |= (1 << x)
        return s

    def try_1_for_2_swap(S_mask, elements_not_in_S):
        """
        Try to find a 1-for-2 swap: remove u, add v, w.
        Returns new S_mask if successful, else None.
        """
        elements_in_S = [i for i in range(1, N + 1) if (S_mask >> i) & 1]
        
        # Shuffle to randomize search order
        random.shuffle(elements_in_S)
        
        for u in elements_in_S:
            # Remove u
            S_temp = S_mask & ~(1 << u)
            
            # Try to add two elements v, w from elements_not_in_S
            # We do a greedy search for v and w
            # First, find a candidate v that can be added to S_temp
            # Then, find a candidate w that can be added to S_temp U {v}
            
            # To be efficient, we can just iterate through candidates
            # If we find any valid pair (v,w), we return immediately
            
            # Optimization: We can pre-filter candidates that don't conflict with S_temp
            # But S_temp changes for each u, so we just check on the fly.
            
            # We want to maximize the chance of finding a pair.
            # Let's iterate v in elements_not_in_S.
            # If v is valid for S_temp, check if there exists w in elements_not_in_S \ {v}
            # such that w is valid for S_temp U {v}.
            
            # This is O(|NotInS|^2) in worst case for each u.
            # |NotInS| is at most 128. 128^2 = 16384.
            # |InS| is at most 128. Total 128 * 16384 ~ 2M ops.
            # This might be too slow if done repeatedly.
            
            # Improvement: Limit the number of v candidates we try.
            # Or, just try the first few valid v's.
            
            found_v = False
            for v in elements_not_in_S:
                if not is_valid_addition(v, S_temp):
                    continue
                
                S_temp_v = S_temp | (1 << v)
                
                # Now find w
                # We can skip v itself
                for w in elements_not_in_S:
                    if w == v:
                        continue
                    if is_valid_addition(w, S_temp_v):
                        # Found a valid swap: remove u, add v, w
                        return S_temp_v | (1 << w)
                
                # If we tried enough v's, break?
                # For now, let's try all v's for the first u, then next u.
                # But this is expensive.
                # Let's add a small limit to the inner loops to keep it fast.
                # Actually, if we find one, we return. If we don't, we move to next u.
                # The cost is high only if we fail.
                # Let's cap the number of w checks per v to avoid long stalls.
                # But if we cap, we might miss a solution.
                # Given N=128, 128*128 is 16k. 128 u's * 16k = 2M.
                # 2M operations in Python is ~0.1-0.2s.
                # If we do this 100 times, that's 20s. Too slow.
                
                # We need a faster heuristic.
                # Instead of checking all w for all v, we can just check a subset.
                # Or, we can stop checking w after the first few valid v's.
                
                # Let's try a different approach:
                # For a fixed u, compute the set of valid additions to S_temp.
                # valid_v = [x for x in elements_not_in_S if is_valid_addition(x, S_temp)]
                # If len(valid_v) >= 2, we might be able to pick two.
                # But they must be compatible with each other too.
                
                # Let's just limit the search depth for w.
                # If we find a v, we check at most 20 w's.
                # This makes it probabilistic but fast.
                
                # Actually, let's just break out of the w loop after the first success.
                # The bottleneck is the nested loops.
                # Let's optimize by breaking early if we find a solution.
                # If we don't find a solution for a specific u, we move to the next u.
                # We should limit the total time spent in this function.
                
                # Let's add a time check inside.
                if time.time() > deadline - 5.0:
                    return None

            # If we didn't return, this u didn't work
            # We don't need to break the u loop, we just try the next u.
            # But if it's taking too long, we should stop.
            if time.time() > deadline - 5.0:
                return None

        return None

    def greedy_polish(S_mask):
        """
        Try to add any remaining elements that don't create bad triples.
        """
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not ((S_mask >> x) & 1):
                    if is_valid_addition(x, S_mask):
                        S_mask |= (1 << x)
                        changed = True
        return S_mask

    # Initial construction
    random.seed(42)
    
    best_set = 0
    best_size = 0
    
    # Generate diverse initial sets
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, 0 - 1)), # Typo in thought, should be range(N, 0, -1)
    ]
    # Correcting order generation
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    for _ in range(5):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        s = greedy_build(order)
        s = greedy_polish(s)
        size = popcount(s)
        if size > best_size:
            best_size = size
            best_set = s

    # Local Search Phase
    current_set = best_set
    current_size = best_size
    
    # We will alternate between:
    # 1. 1-for-2 swaps (to increase size)
    # 2. Simulated Annealing (to escape local minima)
    
    # SA Parameters
    T = 5.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Phase 1: Try 1-for-2 swaps
        # Get list of elements not in current set
        elements_not_in_S = [x for x in range(1, N + 1) if not ((current_set >> x) & 1)]
        
        new_set = try_1_for_2_swap(current_set, elements_not_in_S)
        
        if new_set is not None:
            new_size = popcount(new_set)
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                # If we improved, we can continue trying swaps
                continue
            elif new_size == current_size:
                # Same size, but different structure. Accept with some probability?
                # Or just accept if it looks "better" (hard to define).
                # Let's accept it to explore.
                current_set = new_set
                # Don't update best yet, need to polish
                polished = greedy_polish(current_set)
                if popcount(polished) > current_size:
                    current_set = polished
                    current_size = popcount(current_set)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
                continue
        
        # Phase 2: Simulated Annealing Step
        # Try a random move: Add, Remove, or Swap
        move_type = random.random()
        
        # Define energy: We want to maximize size, minimize bad triples.
        # Since we maintain validity in 1-for-2, we can use a simpler energy for SA:
        # Energy = -size + 10 * bad_triples
        # But calculating bad_triples is expensive.
        # Instead, let's use a "validity-preserving" SA or a "soft" SA.
        # Let's stick to valid moves in SA as well to keep it simple and fast.
        # Move: Swap u (in S) with v (not in S) if valid.
        # Or: Add v if valid.
        # Or: Remove u if it allows adding more later? (Hard to know).
        
        # Let's try: Random valid swap.
        elements_in_S = [x for x in range(1, N + 1) if (current_set >> x) & 1]
        
        if not elements_in_S or not elements_not_in_S:
            break
            
        if move_type < 0.5:
            # Try to add a random element
            if elements_not_in_S:
                v = random.choice(elements_not_in_S)
                if is_valid_addition(v, current_set):
                    new_set = current_set | (1 << v)
                    new_size = current_size + 1
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
        else:
            # Try to swap
            u = random.choice(elements_in_S)
            v = random.choice(elements_not_in_S)
            
            # Check if swapping u for v is valid
            # Remove u, check if v is valid
            S_temp = current_set & ~(1 << u)
            if is_valid_addition(v, S_temp):
                new_set = S_temp | (1 << v)
                # Size is same.
                # Accept with probability based on T?
                # Since size is same, we accept to explore.
                current_set = new_set
                # Try to polish
                polished = greedy_polish(current_set)
                if popcount(polished) > current_size:
                    current_set = polished
                    current_size = popcount(current_set)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set

        T *= cooling_rate
        if T < 0.5:
            T = 0.5

    # Final Polish
    best_set = greedy_polish(best_set)
    
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    if not result:
        return [1]
        
    return result
