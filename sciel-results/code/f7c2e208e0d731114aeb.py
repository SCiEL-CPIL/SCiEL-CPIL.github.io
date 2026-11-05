import time
import numpy as np
import itertools

def _paley_tournament(p):
    """
    Returns orientation list for Paley tournament on p vertices.
    p must be a prime congruent to 3 mod 4.
    Orientation bit for pair (i, j) with i < j:
    0 means i -> j
    1 means j -> i
    """
    # Precompute squares modulo p
    squares = set((x * x) % p for x in range(1, p))
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            # In Paley tournament, i -> j if (j-i) is a quadratic residue mod p
            # Here 0 means i -> j, 1 means j -> i
            if d in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

def _fast_check_property(n, orientation):
    """
    Check if every 4-set has a common dominator.
    Returns True if valid.
    Uses bitmasks for fast domination checks.
    """
    if n > 64:
        # Fallback to slower check for n > 64 (though we expect n=67 to be the target)
        # Actually, let's implement a general check using adjacency list or matrix
        # But for n=67, we can use two 64-bit ints or just numpy.
        # Let's stick to a robust numpy check for safety, but optimized.
        adj = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        
        # Check every 4-set
        for a, b, c, d in itertools.combinations(range(n), 4):
            # Vectorized check for domination
            # a dominates {b,c,d} if adj[a, [b,c,d]] are all 1
            if not (adj[a][b] and adj[a][c] and adj[a][d]):
                if not (adj[b][a] and adj[b][c] and adj[b][d]):
                    if not (adj[c][a] and adj[c][b] and adj[c][d]):
                        if not (adj[d][a] and adj[d][b] and adj[d][c]):
                            return False
        return True
    else:
        # Fast bitmask check for n <= 64
        # out_mask[i] has bit j set if i -> j
        out_mask = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    out_mask[i] |= (1 << j)
                else:
                    out_mask[j] |= (1 << i)
        
        for a, b, c, d in itertools.combinations(range(n), 4):
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            # Check if any vertex in the set dominates the other 3
            # Equivalent to: (out_mask[a] & mask) == mask? No, out_mask[a] contains a? No.
            # out_mask[a] has bits for vertices a beats.
            # We need a to beat b, c, d. So (out_mask[a] & (mask ^ (1<<a))) == (mask ^ (1<<a))
            # Let's just check the 4 candidates.
            if (out_mask[a] & (1<<b)) and (out_mask[a] & (1<<c)) and (out_mask[a] & (1<<d)):
                continue
            if (out_mask[b] & (1<<a)) and (out_mask[b] & (1<<c)) and (out_mask[b] & (1<<d)):
                continue
            if (out_mask[c] & (1<<a)) and (out_mask[c] & (1<<b)) and (out_mask[c] & (1<<d)):
                continue
            if (out_mask[d] & (1<<a)) and (out_mask[d] & (1<<b)) and (out_mask[d] & (1<<c)):
                continue
            return False
        return True

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Try Paley tournaments for primes p ≡ 3 mod 4 in increasing order.
    # Smallest known valid Paley tournament for k=4 is 67.
    # But we check smaller ones just in case (e.g. if knowledge was incomplete or for robustness).
    # Primes p ≡ 3 mod 4: 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, ...
    # n must be >= 5.
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    last_valid = None
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        # Skip if p < 5
        if p < 5:
            continue
            
        result = _paley_tournament(p)
        
        # Verify to be safe.
        if _fast_check_property(p, result['orientation']):
            return result
            
        last_valid = result
        
    # If no candidate worked (should not happen as 67 is known to work), 
    # return the last generated one (which might be invalid if 67 failed, but 67 doesn't fail).
    # To be absolutely safe, if we timed out before checking 67, we might return an invalid one.
    # But 67 is checked early in the list.
    # If we reach here, it means all candidates failed or we timed out.
    # If we timed out, we should probably return the smallest n we *did* generate and check?
    # No, if it's invalid, score is 0.
    # Since 67 is known valid, and it's 5th in the list, and check is fast, we should return it.
    
    # Fallback: return Paley(67) without verification if we ran out of time?
    # Better to assume 67 works.
    if last_valid is None:
         # Emergency fallback
         return _paley_tournament(67)
         
    return last_valid
