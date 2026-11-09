def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights for all 17-bit vectors
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def try_greedy_linear(order, time_limit):
        """
        Try to build a linear code using candidates in the given order.
        Returns the list of codewords.
        """
        current_code = [0]
        basis = []
        target_basis_size = 8  # 2^8 = 256 words

        for g in order:
            if time.time() > time_limit:
                break
            if len(basis) >= target_basis_size:
                break

            # Check if g can be added
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break

            if valid:
                basis.append(g)
                new_words = [c ^ g for c in current_code]
                current_code.extend(new_words)

        return current_code

    # Generate candidate order
    indices = list(range(1, SIZE))
    
    # We will try multiple random shuffles to find a good 256-word linear code
    best_code = []
    best_size = 0
    
    # Phase 1: Find a 256-word linear code
    # Try a few deterministic orders first (e.g., weight sorted)
    # Then random shuffles
    
    # 1. Weight sorted
    indices_weighted = sorted(indices, key=lambda x: -weights[x])
    code = try_greedy_linear(indices_weighted, deadline - 30)
    if len(code) > best_size:
        best_code = code
        best_size = len(code)
        
    if best_size < 256:
        # 2. Random shuffles
        # We have ~30s budget for this phase.
        # Each try takes < 1ms if it fails early, or a bit more if it succeeds.
        # Let's try up to 1000 restarts or until time is up.
        
        for i in range(1000):
            if time.time() > deadline - 30:
                break
            if best_size == 256:
                break
                
            random.shuffle(indices)
            code = try_greedy_linear(indices, deadline - 30)
            if len(code) > best_size:
                best_code = code
                best_size = len(code)
                
    # Phase 2: Non-linear extension
    C = best_code
    if not C:
        return [0] # Fallback
        
    C_set = set(C)
    C_np = np.array(C, dtype=np.int32)
    
    # Find all valid candidates
    valid_candidates = []
    time_limit_phase2 = time.time() + 15 # 15s budget for finding candidates
    
    for w in range(SIZE):
        if time.time() > time_limit_phase2:
            break
        if w in C_set:
            continue
        
        # Check distance to C
        dists = weights[w ^ C_np]
        if dists.min() >= MIN_D:
            valid_candidates.append(w)
            
    # Phase 3: Select maximum subset of valid_candidates
    # We want to add as many as possible.
    # Greedy with random restarts.
    
    best_extension = []
    time_limit_phase3 = deadline - 2
    
    if not valid_candidates:
        return best_code
        
    # We can do many restarts here because it's fast.
    # |V| is likely small (few hundred to few thousand).
    # |S| will be small.
    
    for r in range(200):
        if time.time() > time_limit_phase3:
            break
            
        random.shuffle(valid_candidates)
        current_extension = []
        
        for v in valid_candidates:
            if time.time() > time_limit_phase3:
                break
                
            valid = True
            for c in current_extension:
                if weights[v ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                current_extension.append(v)
                
        if len(current_extension) > len(best_extension):
            best_extension = current_extension
            
    final_code = best_code + best_extension
    return final_code
