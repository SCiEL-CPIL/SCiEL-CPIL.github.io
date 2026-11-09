import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0

    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # Precompute for each divisor d and each position offset p in [0, W-1],
    # whether (L + p) is divisible by d, for all L mod d.
    # For a given window start L and width W, the candidates for divisor i
    # are positions p in [0, W-1] such that (L+p) % divisors[i] == 0.
    
    # For each divisor d, precompute a list of bitmasks for each residue r mod d:
    # mask_d[r] has bit p set iff (r + p) % d == 0, for p in [0, W-1]
    # But W varies, so we compute on the fly per W.
    
    def check_window(L, W):
        """Check if there's a valid assignment in window [L, L+W-1].
        Returns the assignment (list of values) or None."""
        
        # Build candidate bitmasks for each divisor
        # For divisor d, candidate positions p where (L+p) % d == 0
        # p is in [0, W-1]
        # (L+p) % d == 0  <=>  p % d == (-L) % d
        
        candidates = []  # list of bitmasks (integers) for each divisor
        for i, d in enumerate(divisors):
            r = (-L) % d  # p must satisfy p ≡ r (mod d)
            # Build bitmask: bit p set if p % d == r, for p in [0, W-1]
            mask = 0
            # First p >= 0 with p % d == r
            first_p = r
            if first_p >= W:
                return None  # No candidates for this divisor
            for p in range(first_p, W, d):
                mask |= (1 << p)
            candidates.append(mask)
        
        # Now we need to check if we can assign distinct positions to each divisor
        # This is a bipartite matching problem where left side = divisors,
        # right side = positions 0..W-1
        # Use DFS-based matching (Kuhn's algorithm) on the bitmask representation
        
        # match_pos[p] = divisor index, or -1 if unmatched
        match_pos = [-1] * W
        
        def try_kuhn(u, visited):
            """Try to find a matching for divisor u. visited is a bitmask of positions already tried."""
            mask = candidates[u]
            # Iterate through set bits in mask
            m = mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                p = lb.bit_length() - 1
                m ^= lb
                if visited & lb:
                    continue
                visited |= lb
                if match_pos[p] == -1 or try_kuhn(match_pos[p], visited):
                    match_pos[p] = u
                    return True
            return False
        
        # Try to match each divisor
        # Order by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: candidates[i].bit_count())
        
        for u in order:
            visited = 0
            if not try_kuhn(u, visited):
                return None
        
        # Reconstruct assignment
        assignment = [0] * n
        for p in range(W):
            if match_pos[p] != -1:
                assignment[match_pos[p]] = L + p
        
        return assignment
    
    # Iterate span W from minimum possible (n-1) upwards
    for W in range(n - 1, 1000):
        if time.time() > deadline:
            break
        
        # Iterate window start L
        # The pattern repeats every lcm_all, so we only need L in [1, lcm_all]
        # But we also need L >= 1 and L + W - 1 <= MAX_V
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            if L + W - 1 > MAX_V:
                continue
            
            result = check_window(L, W)
            
            if result is not None:
                # Verify the solution
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                return [int(v) for v in result]
    
    # Fallback: Return a valid solution
    return [int(d) for d in divisors]
