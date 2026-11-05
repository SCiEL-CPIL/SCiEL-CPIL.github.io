import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    N = 180
    
    # Precompute conflicts
    # conflict[i] is a set of j such that there exists k with {i,j,k} forbidden
    # Actually, we need to check triples.
    # Let's just check validity of a set directly.
    
    def is_valid(S):
        S_list = sorted(S)
        n = len(S_list)
        # Map value to index for fast lookup? Or just set.
        S_set = set(S)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                # Check if there is c in S such that 1/a = 1/b + 1/c
                # c = ab / (b-a)
                if b == a: continue
                num = a * b
                den = b - a
                if num % den != 0:
                    continue
                c = num // den
                if c in S_set and c != a and c != b:
                    return False
        return True

    # Find best interval [L, 180]
    best_set = set()
    for L in range(180, 0, -1):
        candidate = set(range(L, 181))
        if is_valid(candidate):
            best_set = candidate
            break # This finds the largest L (smallest set)?
            # Wait, we want MAX cardinality.
            # Larger L -> Smaller set.
            # We want the SMALLEST L such that [L, 180] is valid.
            # So we should iterate L from 180 down to 1, and keep the LAST valid one?
            # No, if [L, 180] is valid, [L+1, 180] is also valid.
            # So we want the smallest L.
            # Iterate L from 1 to 180?
            # If [1, 180] is valid, we are done.
            # But it's not.
            # So we iterate L from 1 to 180, find first valid.
            # But checking is_valid is O(N^3) or O(N^2).
            # N=180. N^2 = 32400. Fast.
            
    # Let's re-do:
    best_set = set()
    for L in range(1, 181):
        candidate = set(range(L, 181))
        if is_valid(candidate):
            best_set = candidate
            break
    
    # Now try to add numbers < L
    current_set = set(best_set)
    L_found = min(best_set) if best_set else 181
    
    for x in range(L_found - 1, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        candidate = current_set | {x}
        if is_valid(candidate):
            current_set = candidate
    
    # Local Search / Swap
    # Try to swap a small x not in set with a large y in set
    # This is complex to implement correctly in short time.
    # Let's stick to greedy addition.
    
    return list(current_set)
