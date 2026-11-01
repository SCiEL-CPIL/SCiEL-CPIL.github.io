import time
import random

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases, though task is fixed
        # Simple greedy for general case
        if n == 0:
            return [[]]
        
        # Precompute neighbors for general q, n, radius is complex.
        # For this task, we focus on q=3, n=6, radius=1.
        # If not matching, return a trivial (but likely invalid or large) solution or raise.
        # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
        # and "specialized for q=3, n=6, radius=1".
        # We should handle the specific case efficiently and fallback for others.
        
        # For now, if not 3,6,1, we return a full space solution (valid, size q^n)
        # This is safe but score will be bad if not the target case.
        # But the task is fixed to 3,6,1.
        
        # Let's just return a dummy valid code for other cases to avoid crashes
        # A single codeword is not valid for radius 1 unless n=1 or q=2 etc.
        # Full space is always valid.
        return [list(p) for p in __import__('itertools').product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute neighbors for all 729 words
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = [c]
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.append(c - old_val + new_val)
        neighbors.append(ns)
    
    # Greedy Construction
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        covered = [False] * 729
        code_set = set()
        codes = []
        
        # Initial randomization: pick a random starting codeword?
        # Or just standard greedy.
        
        while True:
            # Check if all covered
            if all(covered):
                break
            
            best_gain = -1
            best_candidate = None
            candidates = []
            
            for c in range(729):
                if c in code_set:
                    continue
                gain = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    candidates = [c]
                elif gain == best_gain and gain > 0:
                    candidates.append(c)
            
            if not candidates:
                # Should not happen if space is connected
                break
                
            # Pick randomly from best candidates to break symmetry
            c = random.choice(candidates)
            code_set.add(c)
            codes.append(c)
            
            for w in neighbors[c]:
                covered[w] = True
                
        return codes

    def local_search(codes, time_limit):
        current_codes = list(set(codes))
        code_set = set(current_codes)
        
        cov = [0] * 729
        # We don't strictly need coverers sets if we just check coverage counts for removal
        # For swap, we need to know if a new codeword covers the exclusive set.
        
        for c in current_codes:
            for w in neighbors[c]:
                cov[w] += 1
        
        best_size = len(current_codes)
        best_codes = current_codes[:]
        
        def remove_code(c):
            for w in neighbors[c]:
                cov[w] -= 1
            code_set.remove(c)
            current_codes.remove(c)
            
        def add_code(c):
            for w in neighbors[c]:
                cov[w] += 1
            code_set.add(c)
            current_codes.append(c)
            
        def is_removable(c):
            for w in neighbors[c]:
                if cov[w] == 1:
                    return False
            return True
            
        def try_removals():
            removed = False
            while True:
                any_removed = False
                for c in current_codes[:]:
                    if is_removable(c):
                        remove_code(c)
                        any_removed = True
                        removed = True
                if not any_removed:
                    break
            return removed

        def try_swaps():
            swapped = False
            # Find codeword with smallest exclusive set
            candidates_for_swap = []
            for c in current_codes:
                exclusive = []
                for w in neighbors[c]:
                    if cov[w] == 1:
                        exclusive.append(w)
                if exclusive:
                    candidates_for_swap.append((len(exclusive), c, exclusive))
                    
            if not candidates_for_swap:
                return False
                
            candidates_for_swap.sort(key=lambda x: x[0])
            
            for size, c, exclusive in candidates_for_swap:
                # Find c' not in code_set that covers all words in exclusive
                found = False
                # Optimization: If exclusive set is large, fewer candidates will cover it.
                # We can iterate through all c' not in set.
                # To speed up, we can precompute for each word w, the list of codewords covering it.
                # But that's heavy.
                # Just iterate 0..728.
                for c_prime in range(729):
                    if c_prime in code_set:
                        continue
                    # Check if c_prime covers all exclusive words
                    ok = True
                    for w in exclusive:
                        if w not in neighbors[c_prime]:
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        remove_code(c)
                        add_code(c_prime)
                        swapped = True
                        found = True
                        break
                if found:
                    break
                    
            return swapped

        # Main Loop
        while time.time() < time_limit:
            if try_removals():
                if len(current_codes) < best_size:
                    best_size = len(current_codes)
                    best_codes = current_codes[:]
                continue
            
            if try_swaps():
                continue
            
            # Stuck, perturb
            # Add random codeword
            attempts = 0
            while attempts < 10:
                r = random.randint(0, 728)
                if r not in code_set:
                    add_code(r)
                    break
                attempts += 1
            else:
                break
                
            try_removals()
            
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                
        return best_codes

    best_global = []
    best_global_size = float('inf')
    
    seed = 42
    while time.time() < deadline:
        # Run greedy with new seed
        codes = greedy_construct(seed)
        seed += 1
        
        # Run local search on this result
        # Allocate remaining time to local search, but ensure we can run multiple iterations
        # If we have > 10s left, give 5s to local search. Else give remaining - 1s.
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        ls_time_limit = min(5.0, time_left - 0.5)
        
        result_codes = local_search(codes, time.time() + ls_time_limit)
        
        if len(result_codes) < best_global_size:
            best_global_size = len(result_codes)
            best_global = result_codes
            
        # Optional: If we found 73, we can stop?
        # The lower bound is 71. 73 is known.
        # If we find 73, it's likely optimal or near.
        # But maybe we can find 71? Unlikely.
        # Let's keep searching to be sure, or stop if 73 found to save time for safety?
        # If we stop at 73, we return 73.
        # If we continue, we might find 73 again or fail to improve.
        # Given 120s limit, and greedy+LS is fast, we can run many restarts.
        
    # Convert best_global back to list of lists
    result = []
    for c in best_global:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
