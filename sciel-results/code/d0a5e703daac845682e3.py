from itertools import product

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = 12
    lcm_period = 27720  # LCM of 1..12
    
    best_m = 0
    best_L = 0
    
    # Order of matching: most constrained first (largest divisor)
    # Divisors are 1..12. Indices 0..11.
    # i=11 is divisor 12. i=0 is divisor 1.
    # We want to match divisor 12 first, then 11, ..., 1.
    # So indices: 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.
    order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    for m in range(0, lcm_period):
        found_L = 0
        start_val = m + 1
        # L must be at least n to fit n distinct integers
        for L in range(n, 145):
            # Check if there exist distinct b_1..b_n in (m, m+L] with i|b_i
            # Bipartite matching
            
            match_to = [-1] * L  # match_to[j] = i_idx (0-11) matched to candidate j, or -1
            
            def try_match(i_idx, visited):
                d = i_idx + 1
                for j in range(L):
                    if visited[j]:
                        continue
                    if (start_val + j) % d != 0:
                        continue
                    visited[j] = True
                    if match_to[j] == -1 or try_match(match_to[j], visited):
                        match_to[j] = i_idx
                        return True
                return False
            
            success = True
            for i_idx in order:
                visited = [False] * L
                if not try_match(i_idx, visited):
                    success = False
                    break
            
            if success:
                found_L = L
                break
        
        if found_L > best_L:
            best_L = found_L
            best_m = m
    
    return best_m
