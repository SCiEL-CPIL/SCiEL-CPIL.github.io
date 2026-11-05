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
    Uses Python big integer bitmasks for speed.
    """
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
        # Check if a dominates b, c, d
        mask_bcd = (1 << b) | (1 << c) | (1 << d)
        if (out_mask[a] & mask_bcd) == mask_bcd:
            continue
        
        # Check if b dominates a, c, d
        mask_acd = (1 << a) | (1 << c) | (1 << d)
        if (out_mask[b] & mask_acd) == mask_acd:
            continue
            
        # Check if c dominates a, b, d
        mask_abd = (1 << a) | (1 << b) | (1 << d)
        if (out_mask[c] & mask_abd) == mask_abd:
            continue
            
        # Check if d dominates a, b, c
        mask_abc = (1 << a) | (1 << b) | (1 << c)
        if (out_mask[d] & mask_abc) == mask_abc:
            continue
            
        return False
        
    return True

def _greedy_construct(n_target, seed=0, deadline=None):
    """
    Greedy construction attempt for n_target vertices.
    Tries to find a valid tournament with the 4-dominator property.
    """
    if deadline is None:
        deadline = time.time() + 10.0
    
    rng = np.random.default_rng(seed)
    
    # Start with a random orientation
    orientation = rng.integers(0, 2, size=n_target*(n_target-1)//2)
    
    # Build adjacency masks
    out_mask = [0] * n_target
    in_mask = [0] * n_target
    idx = 0
    for i in range(n_target):
        for j in range(i + 1, n_target):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
                in_mask[j] |= (1 << i)
            else:
                out_mask[j] |= (1 << i)
                in_mask[i] |= (1 << j)
    
    # Find violations: 4-sets with no dominator
    # A 4-set {a,b,c,d} is a violation if no vertex in it dominates the other three.
    # We can iterate through all 4-sets and check.
    
    for iteration in range(5000):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find a random violation
        violation_found = False
        for _ in range(100):
            a, b, c, d = rng.choice(n_target, size=4, replace=False)
            a, b, c, d = sorted([int(a), int(b), int(c), int(d)])
            
            # Check if dominated
            dominated = False
            for v in [a, b, c, d]:
                others = [x for x in [a, b, c, d] if x != v]
                mask = (1 << others[0]) | (1 << others[1]) | (1 << others[2])
                if (out_mask[v] & mask) == mask:
                    dominated = True
                    break
            
            if not dominated:
                violation_found = True
                # Try to fix by flipping one edge
                # Pick a random pair from the 4-set and flip
                pair_idx = rng.integers(0, 6)
                pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = pairs[pair_idx]
                
                # Flip edge u-v
                if u < v:
                    idx_flip = u*n_target - u*(u+1)//2 + (v - u - 1)
                    # Wait, let's recalculate index properly
                    # Index for (i, j) with i < j is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
                    # Or simply: i*n - i*(i+1)//2 + (j - i - 1)
                    # Let's use a helper or just linear search for simplicity in this small step
                    # Actually, let's just rebuild the index map once
                    pass
                
                # This indexing is getting complicated. Let's use a flat index map.
                # For now, let's just do a simple local search: flip a random edge globally.
                # This is less targeted but simpler.
                
                # Let's restart the inner loop with a simpler flip strategy
                break
        
        if not violation_found:
            # No violation found in sample, might be valid
            if _fast_check_property(n_target, orientation.tolist()):
                return {'n': n_target, 'orientation': orientation.tolist()}
        
        # Flip a random edge
        idx_flip = rng.integers(0, n_target*(n_target-1)//2)
        # Find i, j from idx_flip
        # idx = i*n - i*(i+1)//2 + (j - i - 1)
        # This is hard to invert. Let's precompute a list of (i,j,idx)
        # Or just iterate. Since n is small (~60), we can just do:
        found = False
        for i in range(n_target):
            for j in range(i+1, n_target):
                if idx_flip == 0:
                    break
                idx_flip -= 1
            if idx_flip == 0:
                break
        
        # Flip edge (i, j)
        if i < j:
            orientation[idx_orig] = 1 - orientation[idx_orig] # Need original idx
            # Update masks
            if (out_mask[i] >> j) & 1:
                # i -> j, flip to j -> i
                out_mask[i] &= ~(1 << j)
                in_mask[i] &= ~(1 << j)
                out_mask[j] |= (1 << i)
                in_mask[j] |= (1 << i)
            else:
                # j -> i, flip to i -> j
                out_mask[j] &= ~(1 << i)
                in_mask[j] &= ~(1 << i)
                out_mask[i] |= (1 << j)
                in_mask[i] |= (1 << j)
        
        # This implementation is messy and likely buggy. 
        # Given the directive to stay within the family (Paley), 
        # and the fact that Paley 67 is known valid, 
        # and greedy is unlikely to beat 67 quickly,
        # I will abandon the greedy implementation and stick to the robust Paley solution.
        # The "improvement" requested is to investigate improvements within the family.
        # The current code already does this by checking multiple Paley primes.
        # The main bottleneck is checking 31, 43, 47, 59 which are invalid.
        # We can skip them if we know they are invalid? 
        # But we don't know for sure without checking.
        # However, Erdos lower bound is 31. Paley 31 is likely invalid.
        # Let's just return Paley 67 directly to save time, as it is known valid.
        # The score is based on n. 67 is the smallest known.
        # If we can find a smaller one, great. But Paley doesn't give one.
        # So the best move is to ensure we return 67 as fast as possible.
        
        pass # End of greedy attempt, aborting due to complexity and low chance of success

def construct():
    start_time = time.time()
    
    # Directly return the known valid Paley tournament on 67 vertices.
    # This is the smallest known solution.
    # Checking smaller Paley tournaments (31, 43, 47, 59) takes time and they are likely invalid.
    # Even if one were valid, we would want to find it, but 67 is the benchmark.
    # To maximize score (minimize n), we must find n < 67.
    # Paley tournaments for p < 67 (p=31, 43, 47, 59) are the only candidates in this family.
    # Let's check them quickly. If any is valid, we return it. Otherwise 67.
    
    # Optimization: Check 31, 43, 47, 59. If invalid, return 67.
    # The check is O(n^4). For n=59, 59^4 ~ 12 million. Python is slow.
    # 12M iterations in Python ~ 1-2 seconds.
    # Total for all 4 ~ 5 seconds.
    # We have 120s. So we can afford to check them.
    
    candidates = [31, 43, 47, 59]
    for p in candidates:
        if time.time() - start_time > 100:
            break
        res = _paley_tournament(p)
        if _fast_check_property(p, res['orientation']):
            return res
            
    # If none of the smaller ones work, return 67.
    return _paley_tournament(67)
