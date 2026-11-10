import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030  # LCM(6,10,14,15,21,22,26,33,35,39)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Fallback: the divisors themselves are valid (distinct, positive, <=1e6, each divisible by itself)
    # Span = 39 - 6 = 33
    fallback = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    
    # Minimum possible span is 9 (10 distinct integers need at least span 9)
    # Try increasing window sizes W (span = W)
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        # For each window start L mod LCM, check if we can assign distinct multiples
        # Key optimization: precompute for each divisor the residue class needed
        # For window [L, L+W], divisor d needs a multiple in that range
        # A multiple of d in [L, L+W] exists iff (-L) % d <= W
        # The offsets are: r = (-L) % d, r+d, r+2d, ... <= W
        
        for L in range(LCM):
            if time.time() > deadline:
                break
            
            # Quick filter: check if each divisor has at least one multiple in [L, L+W]
            candidates = []
            valid = True
            for d in divisors:
                r0 = (-L) % d
                if r0 > W:
                    valid = False
                    break
                # Generate all offsets r in [0, W] where (L+r) % d == 0
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Most-constrained-first ordering (MRV heuristic)
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            # Backtracking search for distinct offsets
            used = [False] * (W + 1)
            choices = [-1] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for r in candidates[i]:
                    if not used[r]:
                        used[r] = True
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        used[r] = False
                        choices[i] = -1
                return False
            
            if backtrack(0):
                # Found a valid assignment with span W
                result = [int(L + choices[i]) for i in range(n)]
                # Verify all constraints
                if (len(set(result)) == n and 
                    all(0 < x <= 1000000 for x in result) and
                    all(result[i] % divisors[i] == 0 for i in range(n))):
                    return result
    
    return fallback
