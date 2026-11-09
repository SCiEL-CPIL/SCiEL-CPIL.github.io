import time
import random

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_val = float('inf')
    best_result = None
    
    def get_multiples_in_window(L, W, d):
        """Get all multiples of d in [L, L+W]."""
        if L > MAX_V + W:
            return []
        # First multiple >= L
        k_start = (L + d - 1) // d
        # Last multiple <= L+W
        k_end = (L + W) // d
        if k_start > k_end:
            return []
        # Also ensure <= MAX_V
        max_k = MAX_V // d
        if k_end > max_k:
            k_end = max_k
        if k_start > k_end:
            return []
        return [k * d for k in range(k_start, k_end + 1)]
    
    def backtrack(assigned, used, order, options, idx):
        """Backtracking with MRV already applied via order."""
        if idx == len(order):
            return True
        d = order[idx]
        for val in options[idx]:
            if val not in used:
                assigned[d] = val
                used.add(val)
                if backtrack(assigned, used, order, options, idx + 1):
                    return True
                del assigned[d]
                used.remove(val)
        return False
    
    # Try increasing W
    for W in range(9, 500):
        if time.time() > deadline:
            break
        
        # Try various L positions
        # L should be at least 1, and L+W <= MAX_V
        max_L = MAX_V - W
        if max_L < 1:
            continue
        
        # Try a set of L values: some systematic, some random
        l_candidates = []
        # Systematic: L = 1, 2, ..., min(200, max_L)
        for L in range(1, min(201, max_L + 1)):
            l_candidates.append(L)
        # Also try L values that are multiples of small numbers
        for base in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for mult in range(1, 50):
                L = base * mult
                if 1 <= L <= max_L:
                    l_candidates.append(L)
        # Random L values
        for _ in range(500):
            L = random.randint(1, max_L)
            l_candidates.append(L)
        
        # Deduplicate
        l_candidates = list(set(l_candidates))
        random.shuffle(l_candidates)
        
        for L in l_candidates:
            if time.time() > deadline:
                break
            
            # Get options for each divisor
            options = []
            valid = True
            for d in divisors:
                mults = get_multiples_in_window(L, W, d)
                if not mults:
                    valid = False
                    break
                options.append(mults)
            
            if not valid:
                continue
            
            # Order divisors by number of options (MRV)
            order_idx = sorted(range(n), key=lambda i: len(options[i]))
            order = [divisors[i] for i in order_idx]
            ordered_options = [options[i] for i in order_idx]
            
            # Limit options per divisor to speed up (if too many, sample)
            limited_options = []
            for opts in ordered_options:
                if len(opts) > 20:
                    # Sample up to 20
                    limited_options.append(random.sample(opts, 20))
                else:
                    limited_options.append(opts)
            
            assigned = {}
            used = set()
            if backtrack(assigned, used, order, limited_options, 0):
                # Build result in original order
                result = [int(assigned[d]) for d in divisors]
                val = max(result) - min(result)
                if val < best_val:
                    best_val = val
                    best_result = result
                    if best_val <= W:
                        # Can't do better than W for this W, and we're trying increasing W
                        # Actually we can, but this is a good result
                        break
                break  # Found a solution for this W, move to next W
        
        if best_result is not None and best_val <= W:
            break
    
    if best_result is None:
        # Fallback: just use the divisors themselves
        best_result = divisors[:]
    
    return best_result
