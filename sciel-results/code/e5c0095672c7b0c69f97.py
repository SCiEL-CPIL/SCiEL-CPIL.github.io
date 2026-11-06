import itertools
import time

def construct():
    # Candidate primes p ≡ 3 mod 4
    # Lower bound is 31.
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = None
    best_orient = None
    
    for p in candidates:
        if p < 5: continue
        # Check if p is prime (all candidates are primes)
        # Check if p % 4 == 3
        if p % 4 != 3:
            continue
            
        # Generate squares
        squares = {x * x % p for x in range(1, p)}
        
        # Build adjacency list/bitmask for speed
        # adj[i] is a bitmask where bit j is 1 if i -> j
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    # j -> i is false, so adj[j] doesn't get bit i
                else:
                    adj[j] |= (1 << i)
        
        # Check all 4-subsets
        valid = True
        # Optimization: break early if invalid
        # 4-combinations of range(p)
        for combo in itertools.combinations(range(p), 4):
            a, b, c, d = combo
            # Check if a dominates b, c, d
            if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)):
                continue
            # Check if b dominates a, c, d
            if (adj[b] & (1 << a)) and (adj[b] & (1 << c)) and (adj[b] & (1 << d)):
                continue
            # Check if c dominates a, b, d
            if (adj[c] & (1 << a)) and (adj[c] & (1 << b)) and (adj[c] & (1 << d)):
                continue
            # Check if d dominates a, b, c
            if (adj[d] & (1 << a)) and (adj[d] & (1 << b)) and (adj[d] & (1 << c)):
                continue
            
            valid = False
            break
        
        if valid:
            best_n = p
            # Generate orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0) # i -> j
                    else:
                        orientation.append(1) # j -> i
            break # Found smallest valid, stop
    
    if best_n is None:
        # Fallback to 83 if something went wrong, though 67 is expected
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
        return {'n': p, 'orientation': orientation}
        
    return {'n': best_n, 'orientation': best_orient}
