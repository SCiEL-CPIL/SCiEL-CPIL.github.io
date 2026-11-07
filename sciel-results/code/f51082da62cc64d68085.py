"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Compute LCM of all divisors to define the periodicity
    lcm_all = reduce(_lcm, divisors)
    
    def find_matching(avail):
        """
        Find a perfect matching in the bipartite graph defined by avail.
        avail[i] is the list of candidate values for divisor i.
        Returns the assignment list or None.
        """
        # Map values to indices to handle duplicates in values?
        # Values are distinct in the window? No, values can repeat across divisors.
        # We need to assign distinct values.
        
        # Build adjacency list
        # Left side: divisors 0..n-1
        # Right side: values. Since values can be up to 10^6, we use a dict or list of lists.
        # For small W, values are close. We can just use the values as keys.
        
        # Standard DFS based bipartite matching
        match_r = {}  # value -> divisor index
        result = [None] * n
        
        def dfs(u, seen):
            for v in avail[u]:
                if v in seen:
                    continue
                seen.add(v)
                if v not in match_r or dfs(match_r[v], seen):
                    match_r[v] = u
                    result[u] = v
                    return True
            return False

        # Try to find matching for each divisor
        # Order by size of avail[u] for efficiency
        order = sorted(range(n), key=lambda i: len(avail[i]))
        
        for u in order:
            seen = set()
            if not dfs(u, seen):
                return None
        return result

    def check_feasibility(W, L):
        """
        Check if there is a distinct assignment of multiples in [L, L+W].
        """
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
            
        # Check Hall's condition? 
        # For N=10, just running the matcher is fast enough.
        # If matching fails, return None.
        return find_matching(avail)

    # Binary search for minimum W
    low, high = 1, max_val
    best_result = None
    best_W = max_val + 1
    
    while low <= high:
        mid = (low + high) // 2
        found = False
        for L in range(1, lcm_all + 1):
            if L + mid > max_val:
                break
            result = check_feasibility(mid, L)
            if result is not None:
                # Verify
                if all(isinstance(x, int) for x in result) and \
                   len(set(result)) == n and \
                   all(result[i] % divisors[i] == 0 for i in range(n)) and \
                   all(1 <= x <= max_val for x in result):
                    best_result = result
                    best_W = mid
                    found = True
                    break
        
        if found:
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result:
        return list(best_result)
    
    # Fallback
    return divisors[:]
# EVOLVE-BLOCK-END
