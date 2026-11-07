import time
import sys
import random

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _validate_bitmask(mask):
    """
    Return True if no three distinct a,b,c in the set (represented by bitmask)
    have a+b, a+c, b+c all in the set.
    mask is a Python int with bit i set if (i+1) is in the set (1-indexed).
    """
    # Extract elements
    elems = []
    m = mask
    while m:
        if m & 1:
            elems.append(len(elems) + 1)  # 1-indexed
        m >>= 1
    
    n = len(elems)
    if n < 3:
        return True
    
    # Check all triples
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            s_ab = a + b
            if s_ab > N:
                continue
            if not (mask >> s_ab) & 1:
                continue
            for k in range(j+1, n):
                c = elems[k]
                s_ac = a + c
                s_bc = b + c
                if s_ac <= N and s_bc <= N:
                    if (mask >> s_ac) & 1 and (mask >> s_bc) & 1:
                        return False
    return True

def _can_add_bitmask(mask, x):
    """
    Check if adding x to the set (mask) creates a bad triple.
    x is 1-indexed, not in mask.
    """
    if x > N:
        return False
    
    # A new bad triple must include x.
    # Triple {x, p, q} with p, q in current set, p < q.
    # Sums: x+p, x+q, p+q must all be in new set.
    # Since x is new, x+p, x+q, p+q must be in current set (they can't equal x since elements >= 1).
    
    m = mask
    # Iterate through pairs in current set
    elems = []
    mm = mask
    while mm:
        if mm & 1:
            elems.append(len(elems) + 1)
        mm >>= 1
    
    n = len(elems)
    for i in range(n):
        p = elems[i]
        for j in range(i+1, n):
            q = elems[j]
            s_xp = x + p
            s_xq = x + q
            s_pq = p + q
            if s_xp > N or s_xq > N or s_pq > N:
                continue
            if (mask >> s_xp) & 1 and (mask >> s_xq) & 1 and (mask >> s_pq) & 1:
                return False
    return True

def _max_large_given_small(S_small_mask):
    """
    Given a bitmask for small elements (1..64), find the largest subset of large elements (65..128)
    such that no bad triple exists.
    Returns (mask_large, size_large).
    """
    S_large_mask = 0
    for c in range(65, 129):
        S_large_mask |= (1 << c)
    
    S_small_set = set()
    m = S_small_mask
    while m:
        if m & 1:
            S_small_set.add(len(S_small_set) + 1)
        m >>= 1
    
    # Iterate: remove c from S_large if it forms a bad triple with any pair (a,b) in S_small
    changed = True
    while changed:
        changed = False
        to_remove = []
        for c in range(65, 129):
            if not (S_large_mask >> c) & 1:
                continue
            for a in S_small_set:
                for b in S_small_set:
                    if a == b:
                        continue
                    if a > b:
                        continue
                    s_ab = a + b
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac > N or s_bc > N:
                        continue
                    # Check if s_ab, s_ac, s_bc are all in S = S_small U S_large
                    s_ab_in = ((S_small_mask >> s_ab) & 1) or ((S_large_mask >> s_ab) & 1)
                    s_ac_in = (S_large_mask >> s_ac) & 1
                    s_bc_in = (S_large_mask >> s_bc) & 1
                    if s_ab_in and s_ac_in and s_bc_in:
                        to_remove.append(c)
                        break
                if len(to_remove) > 0 and to_remove[-1] == c:
                    break
            if to_remove and to_remove[-1] == c:
                break
        if to_remove:
            for c in set(to_remove):
                S_large_mask &= ~(1 << c)
            changed = True
    
    return S_large_mask, bin(S_large_mask).count('1')

def _construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(42)
    
    # Start with all large elements
    best_mask = 0
    for c in range(65, 129):
        best_mask |= (1 << c)
    best_size = bin(best_mask).count('1')  # 64
    
    # Try to add small elements greedily
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        small_mask = 0
        large_mask = 0
        for c in range(65, 129):
            large_mask |= (1 << c)
        
        smalls = list(range(1, 65))
        rng.shuffle(smalls)
        
        for x in smalls:
            if time.time() > deadline - 5:
                break
            
            # Check if adding x creates an internal triple in smalls
            new_small_mask = small_mask | (1 << x)
            
            # Check internal triples: x with two existing smalls
            valid = True
            m = small_mask
            elems = []
            mm = small_mask
            while mm:
                if mm & 1:
                    elems.append(len(elems) + 1)
                mm >>= 1
            
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i+1, len(elems)):
                    b = elems[j]
                    s_xa = x + a
                    s_xb = x + b
                    s_ab = a + b
                    if s_xa <= 64 and s_xb <= 64 and s_ab <= 64:
                        if ((new_small_mask >> s_xa) & 1) and ((new_small_mask >> s_xb) & 1) and ((new_small_mask >> s_ab) & 1):
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Calculate new large mask
            new_large_mask, new_large_size = _max_large_given_small(new_small_mask)
            
            new_total = bin(new_small_mask).count('1') + new_large_size
            old_total = bin(small_mask).count('1') + bin(large_mask).count('1')
            
            if new_total > old_total:
                small_mask = new_small_mask
                large_mask = new_large_mask
                total = bin(small_mask).count('1') + bin(large_mask).count('1')
                if total > best_size:
                    best_size = total
                    best_mask = small_mask | large_mask
        
        # Local search: try swaps
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            small_elems = []
            m = small_mask
            mm = small_mask
            while mm:
                if mm & 1:
                    small_elems.append(len(small_elems) + 1)
                mm >>= 1
            
            if not small_elems:
                break
            
            # Pick a random small to remove
            to_remove = rng.choice(small_elems)
            new_small_mask = small_mask & ~(1 << to_remove)
            
            # Try to add a different small
            candidates = [x for x in range(1, 65) if not (small_mask >> x) & 1]
            if not candidates:
                break
            x = rng.choice(candidates)
            
            swap_small_mask = new_small_mask | (1 << x)
            
            # Check internal validity of swap_small_mask
            valid = True
            elems = []
            mm = swap_small_mask
            while mm:
                if mm & 1:
                    elems.append(len(elems) + 1)
                mm >>= 1
            
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i+1, len(elems)):
                    b = elems[j]
                    for k in range(j+1, len(elems)):
                        c = elems[k]
                        s_ab = a + b
                        s_ac = a + c
                        s_bc = b + c
                        if s_ab <= 64 and s_ac <= 64 and s_bc <= 64:
                            if ((swap_small_mask >> s_ab) & 1) and ((swap_small_mask >> s_ac) & 1) and ((swap_small_mask >> s_bc) & 1):
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_large_mask, new_large_size = _max_large_given_small(swap_small_mask)
            new_total = bin(swap_small_mask).count('1') + new_large_size
            old_total = bin(small_mask).count('1') + bin(large_mask).count('1')
            
            if new_total > old_total:
                small_mask = swap_small_mask
                large_mask = new_large_mask
                total = bin(small_mask).count('1') + bin(large_mask).count('1')
                if total > best_size:
                    best_size = total
                    best_mask = small_mask | large_mask

    # Final validation
    if _validate_bitmask(best_mask):
        result = []
        m = best_mask
        while m:
            if m & 1:
                result.append(len(result) + 1)
            m >>= 1
        return result
    else:
        return list(range(65, 129))

def construct():
    return _construct()
