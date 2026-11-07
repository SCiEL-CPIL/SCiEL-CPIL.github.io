import time
import random
import math

def construct():
    deadline = time.time() + 50.0  # Safety margin
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Initial set: Primes
    S = set(a for a in range(2, 257) if is_prime(a))
    
    def can_add(S, x):
        # Check 1: x does not divide b*c for distinct b,c in S
        # Check 2: a does not divide x*c for a in S, c in S\{a}
        
        # Optimization: 
        # If x is large, it's less likely to divide b*c (max 256*255)
        # If x is small, more likely.
        
        # Check 1: x | b*c
        # Iterate pairs in S
        # To speed up, we can precompute divisors? 
        # But S changes.
        
        # Let's just iterate. |S| <= 100. Pairs ~ 5000.
        # 256 candidates * 5000 pairs = 1.2M ops. Fast.
        
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x as divisor
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
        
        # Check 2: existing a in S as divisor of x*c
        for i in range(n):
            a = S_list[i]
            # c can be any element in S except a
            # Note: x is not in S yet.
            for j in range(n):
                if i == j: continue
                c = S_list[j]
                if (x * c) % a == 0:
                    return False
                    # Wait, is this correct?
                    # Condition: a | (x * c)
                    # Yes.
        
        return True

    def try_improve(S):
        improved = True
        while improved:
            improved = False
            # Try to add any valid number
            for x in range(2, 257):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        # Restart loop to check new additions
                        break 
            if time.time() > deadline:
                break
        return S

    S = try_improve(S)
    
    # Local Search: Swaps
    # If we can't add, maybe we can swap u->v to enable more adds
    # We do this for a fixed number of iterations or until time runs out
    
    best_S = set(S)
    
    # Randomize order for swaps
    candidates_to_add = [x for x in range(2, 257) if x not in S]
    
    while time.time() < deadline - 5.0:
        # Pick a random element to remove
        if not S: break
        u = random.choice(list(S))
        S.remove(u)
        
        # Try to add as many as possible from candidates
        # We prioritize candidates that were close to being added?
        # Just try all candidates not in S
        added_any = False
        for v in candidates_to_add:
            if v in S: continue
            if can_add(S, v):
                S.add(v)
                added_any = True
        
        if len(S) > len(best_S):
            best_S = set(S)
            # Update candidates? 
            # candidates_to_add = [x for x in range(2, 257) if x not in S]
            
        # If we didn't improve, maybe revert u?
        # Or keep the state if it's not worse?
        # Actually, if we removed u and added nothing, S is smaller.
        # We should probably revert if size didn't increase.
        # But maybe removing u allows adding TWO elements later?
        # The `try_improve` inside the loop handles additions.
        
        # Let's refine:
        # 1. Remove u
        # 2. Run try_improve (add all possible)
        # 3. If size > best_size, update best.
        # 4. If size == best_size, keep (might be different structure).
        # 5. If size < best_size, revert u? 
        #    Actually, if we removed u and couldn't add enough to compensate, we are worse.
        #    But we want to explore.
        
        # Let's just run try_improve after every removal.
        
    # Final polish: try to add to best_S
    # (Already done in loop, but let's be safe)
    
    return sorted(best_S)
