import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 256
    # Precompute lcm for all pairs? Or just check on the fly?
    # We need to check conflicts with current set S.
    
    # Initial Set: All numbers from 129 to 256
    # We proved these are pairwise compatible.
    S = set(range(129, 257))
    
    # Candidates: 2 to 128
    candidates = list(range(2, 129))
    
    # Precompute conflicts for each candidate with all numbers in [129, 256]
    # conflict[c] = set of a in [129, 256] such that lcm(c, a) <= 256
    conflicts = {}
    for c in candidates:
        conf = set()
        for a in range(129, 257):
            # lcm(c, a) = c * a // gcd(c, a)
            g = math.gcd(c, a)
            lcm_val = (c * a) // g
            if lcm_val <= 256:
                conf.add(a)
        conflicts[c] = conf
    
    # Greedy Swap Improvement
    # Gain(c) = 1/c - sum(1/a for a in conflicts[c] if a in S)
    # We iterate until no improvement or time limit.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_c = None
        best_conf = None
        
        current_time = time.time()
        
        for c in candidates:
            if c in S:
                continue
            conf = conflicts[c]
            # Filter conflicts to those currently in S
            active_conf = [a for a in conf if a in S]
            if not active_conf:
                # If no conflict, we can just add it (gain = 1/c)
                gain = 1.0 / c
            else:
                cost = sum(1.0 / a for a in active_conf)
                gain = 1.0 / c - cost
            
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_c = c
                best_conf = active_conf
        
        if best_c is not None:
            # Apply swap
            S.difference_update(best_conf)
            S.add(best_c)
            improved = True
            
            # Remove best_c from candidates list for next iteration? 
            # No, it might become invalid later if other swaps remove its conflicts? 
            # Actually, if we add c, it might conflict with other candidates? 
            # The conflicts dict is for c vs S (original large numbers). 
            # If we remove some large numbers, c's conflicts with S decrease. 
            # But c might now conflict with OTHER candidates we might add later? 
            # The problem requires pairwise lcm > 256 for ALL pairs in final S. 
            # My conflict check `lcm(c, a) <= 256` was only for a in [129, 256]. 
            # If we add multiple small numbers, they must be compatible with EACH OTHER too. 
            # My greedy swap above only checks compatibility of c with current S. 
            # It does NOT check compatibility of c with other small numbers added previously. 
            # This is a bug. 
            # I need to ensure the final set is valid. 
            # Since I'm adding small numbers, I must check if new c conflicts with existing small numbers in S. 
            # Let's restart with a more robust local search.
            
            break # Restart loop logic to handle small-small conflicts
    
    # Robust Local Search
    # State: S (set of ints)
    # Objective: Sum(1/x for x in S)
    # Constraint: lcm(a,b) > 256 for all a!=b in S
    
    def is_valid(S):
        for i, a in enumerate(S):
            for b in S:
                if a == b: continue
                if math.lcm(a, b) <= 256:
                    return False
        return True
        
    def score(S):
        return sum(1.0 / x for x in S)
        
    # Reset S to a valid state: [129, 256]
    S = set(range(129, 257))
    current_score = score(S)
    
    # Precompute lcm for all pairs in [2, 256] to speed up checks
    # 255 * 255 / 2 = 32k pairs. 
    lcm_cache = {}
    for i in range(2, 257):
        for j in range(i+1, 257):
            lcm_cache[(i,j)] = math.lcm(i,j)
            
    def get_lcm(a, b):
        if a > b: a, b = b, a
        return lcm_cache.get((a,b), math.lcm(a,b))
        
    def is_compatible(x, S):
        for y in S:
            if x == y: continue
            if get_lcm(x, y) <= 256:
                return False
        return True
        
    # Local Search: 1-swap (remove k, add c)
    # And 1-add (add c) if compatible
    # And 1-remove (remove k) if it allows better swaps later? No, removing reduces score.
    
    # We will perform iterative improvement.
    # Candidates for addition: [2, 128]
    # Candidates for removal: S
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # Try all additions
        best_add_gain = 0
        best_add_c = None
        for c in range(2, 129):
            if c in S: continue
            if is_compatible(c, S):
                gain = 1.0 / c
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_c = c
        
        if best_add_c:
            S.add(best_add_c)
            current_score += best_add_gain
            improved = True
            no_improve_count = 0
            continue # Re-evaluate after change
            
        # Try all 1-swaps: remove k from S, add c not in S
        # Gain = 1/c - 1/k
        # Constraint: c compatible with S \ {k}
        best_swap_gain = 0
        best_swap_c = None
        best_swap_k = None
        
        # Optimization: Only consider swaps where 1/c > 1/k, i.e., c < k
        for k in list(S):
            # If k is small, 1/k is large, hard to beat. 
            # But k might be blocking many c's.
            # We check if there exists c < k such that c is compatible with S \ {k}
            # And 1/c - 1/k > best_swap_gain
            
            # Heuristic: If 1/max(S) - 1/k <= best_swap_gain, skip? 
            # Max possible gain from adding any c<k is 1/k - 1/k (if c=k, not allowed) 
            # Actually max c is k-1. Gain < 1/(k-1) - 1/k = 1/(k(k-1)).
            # This is small for large k. 
            # For small k, gain is larger.
            
            # Iterate c
            for c in range(2, k):
                if c in S: continue
                # Check compatibility with S \ {k}
                compat = True
                for y in S:
                    if y == k: continue
                    if get_lcm(c, y) <= 256:
                        compat = False
                        break
                if compat:
                    gain = 1.0 / c - 1.0 / k
                    if gain > best_swap_gain + 1e-12:
                        best_swap_gain = gain
                        best_swap_c = c
                        best_swap_k = k
                        # Early break? No, there might be better c.
                        
        if best_swap_c:
            S.discard(best_swap_k)
            S.add(best_swap_c)
            current_score += best_swap_gain
            improved = True
            no_improve_count = 0
            continue
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    # Final validation
    if not is_valid(S):
        # Fallback to safe set
        return list(range(129, 257))
        
    return list(S)

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# To fit the format, I will wrap it.
# EVOLVE-BLOCK-END
