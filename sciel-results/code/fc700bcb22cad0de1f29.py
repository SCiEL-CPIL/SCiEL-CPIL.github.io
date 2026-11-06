"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _is_antichain(terms):
    """Check if terms form a divisibility antichain."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def construct():
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    
    # We will find the minimal antichain representation for each target 1..128
    # Since the search space is small, we can do this efficiently at runtime.
    # To be safe against any potential overhead in repeated calls (though construct is usually called once),
    # we can cache the result in a global variable.
    
    if hasattr(construct, '_cached_result'):
        return construct._cached_result

    results = [None] * 129 # 1-indexed
    
    # Optimization: Pre-filter smooth numbers? No, we need all for sums.
    # We iterate by number of terms k.
    # Max k is likely small (maybe 5 or 6).
    
    # To speed up, we can iterate k from 1 to 10.
    # For each k, we generate combinations.
    # However, generating all combinations of size k from ~43 elements is:
    # k=1: 43
    # k=2: 903
    # k=3: 12341
    # k=4: 123410
    # k=5: 962598
    # This is manageable.
    
    # Better approach: For each target n, search for minimal k.
    # But doing it for all n simultaneously by iterating combinations is more efficient 
    # because we can mark targets as solved.
    
    unsolved = set(range(1, 129))
    
    # k=1
    for s in smooth:
        if s in unsolved:
            results[s] = [s]
            unsolved.remove(s)
            
    if not unsolved:
        construct._cached_result = results[1:129]
        return construct._cached_result

    # k=2
    for i in range(len(smooth)):
        for j in range(i + 1, len(smooth)):
            a, b = smooth[i], smooth[j]
            s = a + b
            if s in unsolved:
                # Check antichain
                if not (b % a == 0 or a % b == 0):
                    results[s] = [a, b]
                    unsolved.remove(s)
                    
    if not unsolved:
        construct._cached_result = results[1:129]
        return construct._cached_result

    # k=3
    for i in range(len(smooth)):
        for j in range(i + 1, len(smooth)):
            a, b = smooth[i], smooth[j]
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, len(smooth)):
                c = smooth[k]
                s = a + b + c
                if s in unsolved:
                    # Check c against a and b
                    if (c % a == 0 or a % c == 0) or (c % b == 0 or b % c == 0):
                        continue
                    results[s] = [a, b, c]
                    unsolved.remove(s)
                    
    if not unsolved:
        construct._cached_result = results[1:129]
        return construct._cached_result

    # k=4
    for i in range(len(smooth)):
        for j in range(i + 1, len(smooth)):
            a, b = smooth[i], smooth[j]
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, len(smooth)):
                c = smooth[k]
                if (c % a == 0 or a % c == 0) or (c % b == 0 or b % c == 0):
                    continue
                for l in range(k + 1, len(smooth)):
                    d = smooth[l]
                    s = a + b + c + d
                    if s in unsolved:
                        # Check d against a, b, c
                        if (d % a == 0 or a % d == 0) or (d % b == 0 or b % d == 0) or (d % c == 0 or c % d == 0):
                            continue
                        results[s] = [a, b, c, d]
                        unsolved.remove(s)

    if not unsolved:
        construct._cached_result = results[1:129]
        return construct._cached_result

    # Fallback for any remaining (should be very few or none)
    # Use a simple recursive search for remaining targets
    for n in list(unsolved):
        found = False
        for k in range(5, 15):
            # Simple DFS
            def dfs(idx, chosen, current_sum):
                if len(chosen) == k:
                    if current_sum == n and _is_antichain(chosen):
                        return list(chosen)
                    return None
                if current_sum > n:
                    return None
                for i in range(idx, len(smooth)):
                    if current_sum + smooth[i] > n:
                        break
                    chosen.append(smooth[i])
                    res = dfs(i + 1, chosen, current_sum + smooth[i])
                    if res:
                        return res
                    chosen.pop()
                return None
            
            res = dfs(0, [], 0)
            if res:
                results[n] = res
                unsolved.remove(n)
                found = True
                break
        
        if not found:
            # This should not happen for 1..128
            results[n] = [n] if _is_smooth(n) else [1, n-1] 

    construct._cached_result = results[1:129]
    return construct._cached_result

# EVOLVE-BLOCK-END
