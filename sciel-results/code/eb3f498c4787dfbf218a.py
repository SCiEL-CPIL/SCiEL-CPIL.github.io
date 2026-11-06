import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = 512
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # 1. Greedy Initialization
    chosen = []
    for a in range(N - 1, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    def get_score(s):
        return sum(weights[a] for a in s)
    
    best_set = list(chosen)
    best_score = get_score(best_set)
    
    # 2. Local Search
    # Precompute coprime check for speed? 
    # Given small N, math.gcd is fast enough.
    
    def is_coprime_to_set(a, s):
        for b in s:
            if math.gcd(a, b) != 1:
                return False
        return True
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add a number
        # Find the best a not in best_set that is coprime to all in best_set
        best_add = None
        best_add_w = 0
        for a in range(N - 1, 0, -1):
            if a in best_set:
                continue
            if is_coprime_to_set(a, best_set):
                if weights[a] > best_add_w:
                    best_add_w = weights[a]
                    best_add = a
                    # Since we iterate from high to low, the first one found is the best
                    break
        
        if best_add is not None:
            new_set = list(best_set)
            new_set.append(best_add)
            new_score = get_score(new_set)
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
                improved = True
                continue # Restart loop as set changed
            
        # Try to swap a number a in best_set with b not in best_set
        # w_b > w_a
        for i, a in enumerate(best_set):
            # Try to replace a with some b
            # b must be coprime to best_set \ {a}
            # and w_b > w_a
            # We want the best such b
            best_swap_b = None
            best_swap_w = 0
            for b in range(N - 1, 0, -1):
                if b in best_set:
                    continue
                if weights[b] <= weights[a]:
                    break # Since weights are decreasing, no better b exists
                # Check if b is coprime to all in best_set except a
                ok = True
                for j, c in enumerate(best_set):
                    if j == i:
                        continue
                    if math.gcd(b, c) != 1:
                        ok = False
                        break
                if ok:
                    best_swap_b = b
                    best_swap_w = weights[b]
                    break # First one is best
            
            if best_swap_b is not None:
                new_set = list(best_set)
                new_set[i] = best_swap_b
                new_score = get_score(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
                    break # Restart loop as set changed
        
        if improved:
            continue
            
        # Try to replace one number a with two numbers b, c
        # w_b + w_c > w_a
        # b, c coprime to each other and to best_set \ {a}
        # This is more complex. 
        # Given the small size, we can try a limited search.
        # For each a in best_set:
        #   Find top 2 candidates b, c not in best_set, coprime to best_set\{a} and each other.
        #   If w_b + w_c > w_a, replace.
        
        for i, a in enumerate(best_set):
            rest = [x for j, x in enumerate(best_set) if j != i]
            candidates = []
            for b in range(N - 1, 0, -1):
                if b in best_set:
                    continue
                if is_coprime_to_set(b, rest):
                    candidates.append(b)
                if len(candidates) >= 10: # Limit candidates for speed
                    break
            
            # Check pairs in candidates
            found_pair = False
            for k in range(len(candidates)):
                b = candidates[k]
                for l in range(k + 1, len(candidates)):
                    c = candidates[l]
                    if math.gcd(b, c) == 1:
                        if weights[b] + weights[c] > weights[a]:
                            new_set = list(best_set)
                            new_set[i] = b
                            new_set.append(c)
                            new_score = get_score(new_set)
                            if new_score > best_score:
                                best_set = new_set
                                best_score = new_score
                                improved = True
                                found_pair = True
                                break
                if found_pair:
                    break
            if found_pair:
                break # Restart loop as set changed
                
    return best_set
