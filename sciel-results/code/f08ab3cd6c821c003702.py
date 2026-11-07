def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    # Precompute squarefree kernels
    sf_cache = {}
    for i in range(211, 10211):
        sf_cache[i] = squarefree_kernel(i)
        
    # Find best pair
    kernel_to_numbers = defaultdict(list)
    for i in range(211, 10211):
        kernel_to_numbers[sf_cache[i]].append(i)
        
    best_pair = None
    best_pair_max = 10210
    
    # To speed up pair search, we can iterate a and look for b.
    # We want min max(a, b).
    # Sort candidates.
    candidates = list(range(211, 10211))
    
    for a in candidates:
        if time.time() > deadline:
            break
        target = squarefree_kernel(210 * a)
        if target in kernel_to_numbers:
            # Find smallest b != a
            for b in kernel_to_numbers[target]:
                if b != a:
                    curr_max = max(a, b)
                    if curr_max < best_pair_max:
                        best_pair_max = curr_max
                        best_pair = (a, b)
                    break
                    
    # Find best triple with max < best_pair_max
    best_triple = None
    best_triple_max = best_pair_max
    
    # We only care about M < best_pair_max
    # If best_pair_max is 240, we check M up to 239.
    limit_M = best_pair_max - 1
    
    # We will iterate M from 211 to limit_M
    # Maintain kernel_to_nums for current range
    current_kernel_to_nums = defaultdict(list)
    
    # We need to check all pairs in {211, ..., M}
    # To be efficient, we can just re-check or update.
    # Given small N, re-checking is fine.
    
    for M in range(211, limit_M + 1):
        if time.time() > deadline:
            break
            
        # Add M to current set
        current_kernel_to_nums[sf_cache[M]].append(M)
        
        # Check all pairs a, b in {211, ..., M}
        # To avoid O(N^3) with large constant, we can just iterate a < b <= M
        # But we only need to find ONE valid triple for this M.
        # If found, return immediately as it is the global min max.
        
        found = False
        # Iterate a from 211 to M
        for a in range(211, M + 1):
            if found: break
            for b in range(a + 1, M + 1):
                # Compute target kernel for c
                target_c = squarefree_kernel(210 * a * b)
                if target_c in current_kernel_to_nums:
                    # Check if there is c != a, c != b
                    for c in current_kernel_to_nums[target_c]:
                        if c != a and c != b:
                            # Found a triple with max M
                            best_triple = (a, b, c)
                            best_triple_max = M
                            found = True
                            break
            if found: break
            
        if found:
            # Since we iterate M increasing, the first found is the minimum max
            break

    # Decide which to return
    if best_triple is not None:
        return list(best_triple)
    else:
        if best_pair is not None:
            return list(best_pair)
        else:
            # Fallback
            return [840]
