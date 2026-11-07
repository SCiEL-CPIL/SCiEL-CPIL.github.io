"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _is_valid(S):
    """Check if set S has no three distinct members a,b,c with all pair sums in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    return False
    return True

def _greedy_with_order(order, n):
    """Build a set by trying elements in given order, skipping if it creates a violation.
    Uses bitmask for fast membership checks.
    """
    S_mask = 0
    S_set = set() # Keep set for iteration if needed, but mask is primary
    
    for x in order:
        if (S_mask >> x) & 1:
            continue
            
        # Tentatively add x
        S_mask_new = S_mask | (1 << x)
        
        # Check validity
        # We need to check if there exist a, b in S_mask (without x) 
        # such that a+x in S_mask_new, b+x in S_mask_new, and a+b in S_mask_new.
        # Note: a+b must be in S_mask_new. Since a,b < x is not guaranteed, 
        # but a,b are in S_mask (old), a+b could be x? No, a,b distinct from x.
        # Actually, a+b could be x? If a+b=x, then a+b is in S_mask_new.
        
        # Efficient check:
        # Iterate over a in S_set (old S)
        # For each a, check if a+x is in S_mask_new.
        # If yes, let b_candidates be elements b in S_set (old S), b != a, such that b+x in S_mask_new.
        # Then check if a+b is in S_mask_new.
        
        valid = True
        # We can iterate over the bits of S_mask
        # But S_set is easier to iterate over if we maintain it.
        # However, maintaining S_set adds overhead. 
        # Let's just iterate over the list of elements in S_mask.
        # Since N=128, we can just check all a in 1..128.
        
        # Optimization: 
        # If we add x, the new sums involving x are a+x.
        # A violation is a triple {a,b,x} where a,b in S_old.
        # Conditions:
        # 1. a+x in S_new
        # 2. b+x in S_new
        # 3. a+b in S_new
        
        # Let A = { a in S_old | a+x in S_new }
        # For each a in A, we need to find b in A \ {a} such that a+b in S_new.
        
        # To avoid list(S_mask) overhead, we can just loop 1..n
        # But 128 iterations is fine.
        
        # Actually, we can precompute A once per x.
        A = []
        # Iterate through all possible a in 1..n
        # We can use bit tricks or just loop
        for a in range(1, n + 1):
            if (S_mask >> a) & 1: # a in S_old
                if (S_mask_new >> (a + x)) & 1: # a+x in S_new
                    A.append(a)
        
        # Now check pairs in A
        for i in range(len(A)):
            a = A[i]
            for j in range(i + 1, len(A)):
                b = A[j]
                if (S_mask_new >> (a + b)) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask = S_mask_new
            S_set.add(x)
            
    return S_set

def construct():
    n = P['n']
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Baseline: {63, ..., 128} has size 66
    best_set = set(range(63, n + 1))
    best_size = len(best_set)
    
    # Try randomized greedy approaches
    rng = random.Random(42)
    
    # Strategy 1: Try to extend the upper half by including some lower elements
    # Start from upper half and try to add elements from below
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Random ordering biased toward upper half
        upper = list(range(63, n + 1))
        lower = list(range(1, 63))
        rng.shuffle(upper)
        rng.shuffle(lower)
        
        # Mix: put upper half first (to maintain base), then try some lower
        mix_ratio = rng.random()
        num_lower = int(len(lower) * mix_ratio)
        order = upper + lower[:num_lower]
        
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Pure random orders
    for trial in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Start from a random subset of the upper half, then greedily add
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Randomly pick a subset of upper half to start
        upper = list(range(63, n + 1))
        rng.shuffle(upper)
        start_size = rng.randint(30, 60)
        start_set = set(upper[:start_size])
        
        # Greedily add remaining elements in random order
        remaining = [x for x in range(1, n + 1) if x not in start_set]
        rng.shuffle(remaining)
        
        S = _greedy_with_order(start_set.union(remaining), n) # Pass combined order? 
        # Wait, _greedy_with_order takes an order. 
        # If we pass start_set elements first, it will add them. 
        # But _greedy_with_order starts with empty set. 
        # So we should pass an order where start_set elements are first.
        
        # Let's construct order: start_set elements first, then remaining
        order = list(start_set) + remaining
        S = _greedy_with_order(order, n)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    if time.time() > deadline - 2.0:
        # Return best found
        return [int(x) for x in sorted(best_set)]
    
    # Final verification
    if _is_valid(best_set):
        return [int(x) for x in sorted(best_set)]
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, n + 1)]
# EVOLVE-BLOCK-END
