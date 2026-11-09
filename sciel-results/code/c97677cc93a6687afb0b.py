def construct():
    import time
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    from math import gcd

    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if gcd(S_list[i], S_list[j]) > 1:
                    return False
        return True

    def local_search(S):
        S = list(S)
        S_set = set(S)
        current_sum = sum(S)
        
        # Pre-sort candidates descending for refill
        all_candidates = list(range(N, 0, -1))
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add elements (Greedy Add)
            # We can just iterate through all_candidates and add if possible
            # But to be efficient, we can just check if any can be added.
            # Since we do this repeatedly, we can break early if no add is found in a pass?
            # Actually, just do one pass of adds.
            added_any = False
            for v in all_candidates:
                if v in S_set:
                    continue
                ok = True
                for u in S:
                    if gcd(v, u) > 1:
                        ok = False
                        break
                if ok:
                    S.append(v)
                    S_set.add(v)
                    current_sum += v
                    added_any = True
                    improved = True
            if not added_any:
                # If no adds, try remove-refill
                for u in list(S): # Iterate over a copy
                    if time.time() > deadline - 2:
                        break
                    # Try removing u
                    S.remove(u)
                    S_set.remove(u)
                    old_sum = current_sum - u
                    current_sum = old_sum
                    
                    # Refill: Add largest compatible numbers
                    refill_sum = 0
                    new_elements = []
                    for v in all_candidates:
                        if v in S_set:
                            continue
                        ok = True
                        for w in S:
                            if gcd(v, w) > 1:
                                ok = False
                                break
                        if ok:
                            S.append(v)
                            S_set.add(v)
                            current_sum += v
                            refill_sum += v
                            new_elements.append(v)
                    
                    # Check if this move improved the sum
                    # The move is valid if we replaced u with new_elements.
                    # Gain = refill_sum - u
                    if refill_sum > u:
                        improved = True
                    else:
                        # Revert
                        for v in new_elements:
                            S.remove(v)
                            S_set.remove(v)
                        current_sum -= refill_sum
                        S.append(u)
                        S_set.add(u)
                        current_sum += u
                        # If no improvement, we might be stuck.
                        # But we tried all u's.
        
        return S

    # 1. Try ILP
    result = None
    try:
        import scipy.optimize as opt
        import numpy as np
        from scipy.sparse import csc_matrix
        
        A_rows = []
        A_cols = []
        A_vals = []
        
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                if gcd(i, j) > 1:
                    idx = len(A_rows) // 2
                    A_rows.append(idx)
                    A_cols.append(i-1)
                    A_vals.append(1)
                    
                    A_rows.append(idx)
                    A_cols.append(j-1)
                    A_vals.append(1)
        
        A = csc_matrix((A_vals, (A_rows, A_cols)), shape=(len(A_rows)//2, N))
        b = np.ones(len(A_rows)//2)
        
        c = -np.arange(1, N+1, dtype=float)
        bounds = [(0, 1)] * N
        integrality = np.ones(N)
        
        # Use remaining time for ILP, but limit it to ensure local search runs
        # If ILP takes too long, it's better to have a good heuristic result.
        remaining_time = max(1, int(deadline - time.time()) - 10)
        res = opt.milp(c, constraints=(A, b), bounds=bounds, integrality=integrality, options={'time_limit': remaining_time})
        
        # Accept if optimal (0) or feasible solution found (1, 2, etc. if x is not None)
        if res.x is not None:
            x = res.x
            # Round to nearest integer, check validity
            x_int = [1 if val > 0.5 else 0 for val in x]
            result = [int(i+1) for i, val in enumerate(x_int) if val == 1]
            
            # Verify validity
            if is_valid_set(result):
                # Ensure 1 is included if possible? 1 is coprime to all.
                if 1 not in result:
                    result.append(1)
            else:
                result = None
    except Exception:
        result = None
    
    # 2. Fallback: Prime Powers
    if result is None:
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0:
                    return False
            return True
            
        primes = [a for a in range(2, N+1) if is_prime(a)]
        result = [1]
        for p in primes:
            max_p = p
            while max_p * p <= N:
                max_p *= p
            result.append(max_p)
            
    # 3. Local Search Refinement
    if time.time() < deadline - 2:
        result = local_search(result)
        
    # Final check and type conversion
    final_result = []
    for v in result:
        if isinstance(v, bool):
            continue
        if not isinstance(v, int):
            continue
        final_result.append(int(v))
        
    return final_result
