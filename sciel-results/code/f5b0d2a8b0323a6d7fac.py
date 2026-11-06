import time
import random
from math import gcd

N = 128
VALS = list(range(1, N + 1))

def _ensure_pre():
    global _GCD, _CONFLICTS
    if '_GCD' in globals():
        return
    
    # _GCD[i][j] = gcd(VALS[i], VALS[j])
    _GCD = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            _GCD[i][j] = gcd(VALS[i], VALS[j])
            
    # _CONFLICTS[i] is a list of sets? 
    # Better: For each element x, we want to know which pairs (i,j) in the current set S 
    # would form a bad triple with x.
    # A triple (x, i, j) is bad if gcd(x,i) == gcd(x,j) == gcd(i,j).
    # Let's precompute for each x, a list of (i, j) pairs such that (x,i,j) is a bad triple.
    # _BAD_TRIPLES[x] = list of (i, j) indices.
    # Then when adding x to S, we count how many pairs (i,j) from _BAD_TRIPLES[x] are both in S.
    
    _BAD_TRIPLES = [[] for _ in range(N)]
    for x in range(N):
        for i in range(N):
            if i == x: continue
            for j in range(N):
                if j == x or j == i: continue
                # Check if (x, i, j) is a bad triple
                g1 = _GCD[x][i]
                g2 = _GCD[x][j]
                g3 = _GCD[i][j]
                if g1 == g2 == g3:
                    # To avoid duplicates in counting, we can store unique pairs (min(i,j), max(i,j))
                    # But for efficiency in checking "is i in S and j in S", we just need the indices.
                    # We will store (i, j) with i < j to keep it consistent?
                    # Actually, the check will be: for (i,j) in _BAD_TRIPLES[x]: if (S>>i)&1 and (S>>j)&1: count++.
                    # It doesn't matter if we have duplicates or order, as long as we count correctly.
                    # However, storing i<j reduces list size slightly? No, (i,j) and (j,i) are distinct in loop.
                    # We can just store i, j.
                    _BAD_TRIPLES[x].append((i, j))
                    
    _CONFLICTS = _BAD_TRIPLES

def _delta_add(mask, x):
    """Calculate increase in violations if x is added to mask."""
    if (mask >> x) & 1:
        return 0
    count = 0
    for i, j in _CONFLICTS[x]:
        if ((mask >> i) & 1) and ((mask >> j) & 1):
            count += 1
    return count

def _delta_remove(mask, x):
    """Calculate decrease in violations if x is removed from mask."""
    if not ((mask >> x) & 1):
        return 0
    count = 0
    for i, j in _CONFLICTS[x]:
        if ((mask >> i) & 1) and ((mask >> j) & 1):
            count += 1
    return count

def _violations_count(mask):
    """Total violations in mask."""
    # Sum of deltas for all elements? No, that double counts.
    # Each bad triple (a,b,c) is counted in _CONFLICTS[a] (as b,c), _CONFLICTS[b] (as a,c), _CONFLICTS[c] (as a,b).
    # So total violations = (sum of _delta_remove for all x in mask) / 3.
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += _delta_remove(mask, x)
    return total // 3

def _popcount(m):
    return bin(m).count('1')

def _greedy_construct():
    """
    Greedy construction.
    We try to add elements in an order that minimizes conflicts.
    Elements that are "isolated" or have few conflicts are better to add early?
    Or elements that are "central" (many conflicts) should be avoided?
    Actually, we want to maximize size.
    Let's try sorting by number of conflicts (degree in conflict hypergraph) ascending.
    """
    # Calculate static conflict count (potential conflicts with any other pair)
    # This is just len(_CONFLICTS[x])
    order = sorted(range(N), key=lambda x: len(_CONFLICTS[x]))
    
    mask = 0
    for x in order:
        if _delta_add(mask, x) == 0:
            mask |= (1 << x)
    return mask

def _local_search(mask, deadline):
    """
    Local search to improve the mask.
    Objective: Maximize size, subject to 0 violations.
    If violations > 0, minimize violations first.
    """
    s = _popcount(mask)
    v = _violations_count(mask)
    
    best_mask = mask
    best_s = s
    best_v = v
    
    # If we start with violations, we need to remove them.
    # If we start with 0 violations, we try to add.
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(N) if not ((mask >> x) & 1)]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            dv = _delta_add(mask, x)
            new_mask = mask | (1 << x)
            new_s = s + 1
            new_v = v + dv
            
            # Accept if:
            # 1. Improves validity (new_v < v)
            # 2. Same validity, improves size (new_v == v and new_s > s)
            # 3. Worse validity, but accepted with prob exp(-delta/T)
            # Note: We prioritize 0 violations.
            # If v > 0, we want to reduce v.
            # If v == 0, we want to increase s.
            
            if new_v < v:
                accept = True
            elif new_v == v:
                if new_s > s:
                    accept = True
                else:
                    accept = False # Same size, same v, no reason to move
            else:
                # new_v > v
                # Only accept if we are stuck and want to explore?
                # Or if the increase in v is small and T is high.
                # But we want to stay valid if possible.
                # Let's only accept worse v if v is already 0? No, that breaks validity.
                # Actually, if v > 0, we MUST reduce v.
                # If v == 0, we MUST keep v == 0 or accept small penalty?
                # Standard SA: accept with prob exp(-(new_v - v)/T)
                # But if v=0, new_v=1 is a big jump.
                # Let's use a penalty score.
                # Score = s - 100*v.
                # Delta Score = new_s - s - 100*(new_v - v)
                # Accept if Delta Score > 0 or prob exp(Delta Score / T)
                
                # Let's stick to strict validity first.
                # If v == 0, only accept if new_v == 0.
                # If v > 0, accept if new_v < v, or (new_v == v and random < exp(1/T)) to escape plateaus?
                # Let's try: if v > 0, accept if new_v <= v.
                # if v == 0, accept if new_v == 0.
                
                if v == 0:
                    accept = False
                else:
                    # Allow some worsening to escape local minima in violation space
                    accept = random.random() < (2.718281828 ** (-(new_v - v) / T))
            
            if accept:
                mask, s, v = new_mask, new_s, new_v
                
        elif move_type < 0.8:
            # Remove
            candidates = [x for x in range(N) if (mask >> x) & 1]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            dv = _delta_remove(mask, x)
            new_mask = mask & ~(1 << x)
            new_s = s - 1
            new_v = v - dv
            
            # Removing always reduces or keeps v same (dv >= 0).
            # new_v <= v.
            # If v > 0, this is good.
            # If v == 0, this reduces size, so only accept if it allows adding more later?
            # In SA, we accept if it improves score.
            # Score = s - 100*v.
            # Delta = -1 - 100*(-dv) = -1 + 100*dv.
            # If dv >= 1, delta >= 99 > 0. Always accept if it removes a violation.
            # If dv == 0, delta = -1. Accept with prob exp(-1/T).
            
            if v > 0:
                if new_v < v:
                    accept = True
                else:
                    accept = random.random() < (2.718281828 ** (-1 / T))
            else:
                # v == 0, new_v == 0
                # Size decreases. Accept with low prob to allow exploration.
                accept = random.random() < (2.718281828 ** (-1 / T))
            
            if accept:
                mask, s, v = new_mask, new_s, new_v
                
        else:
            # Swap
            remove_cands = [x for x in range(N) if (mask >> x) & 1]
            add_cands = [x for x in range(N) if not ((mask >> x) & 1)]
            if not remove_cands or not add_cands:
                T *= cooling
                continue
            r = random.choice(remove_cands)
            a = random.choice(add_cands)
            
            # Calculate delta
            # Remove r: dv_r = _delta_remove(mask, r)
            # Add a to (mask \ {r}): dv_a = _delta_add(mask & ~(1<<r), a)
            # new_v = v - dv_r + dv_a
            # new_s = s
            
            m_temp = mask & ~(1 << r)
            dv_r = _delta_remove(mask, r)
            dv_a = _delta_add(m_temp, a)
            
            new_v = v - dv_r + dv_a
            new_s = s
            
            # Acceptance
            if new_v < v:
                accept = True
            elif new_v == v:
                # Size same. Accept with some prob to shuffle?
                accept = random.random() < 0.1
            else:
                if v == 0:
                    accept = False
                else:
                    accept = random.random() < (2.718281828 ** (-(new_v - v) / T))
            
            if accept:
                mask = m_temp | (1 << a)
                v = new_v
                # s unchanged
        
        # Track best
        # We prefer valid sets (v=0) with max size.
        # If v>0, we prefer lower v.
        if v < best_v:
            best_mask, best_v = mask, v
            best_s = s
        elif v == best_v:
            if s > best_s:
                best_mask, best_s = mask, s
                
        T *= cooling
        
    return best_mask

def construct():
    start = time.time()
    deadline = start + 50.0 # Safety margin
    
    _ensure_pre()
    
    # Multiple starts
    best_mask = 0
    best_size = 0
    best_violations = float('inf')
    
    # Start 1: Greedy
    m1 = _greedy_construct()
    m1_final = _local_search(m1, deadline - 20.0)
    s1 = _popcount(m1_final)
    v1 = _violations_count(m1_final)
    if v1 < best_violations or (v1 == best_violations and s1 > best_size):
        best_mask = m1_final
        best_size = s1
        best_violations = v1
        
    # Start 2: Randomized Greedy
    for _ in range(5):
        # Shuffle order
        order = list(range(N))
        random.shuffle(order)
        # Sort by conflict count with some noise?
        # Just use shuffled order for diversity
        mask = 0
        for x in order:
            if _delta_add(mask, x) == 0:
                mask |= (1 << x)
        
        m2 = _local_search(mask, deadline - 10.0)
        s2 = _popcount(m2)
        v2 = _violations_count(m2)
        if v2 < best_violations or (v2 == best_violations and s2 > best_size):
            best_mask = m2
            best_size = s2
            best_violations = v2
            
    # Final Polish
    if time.time() < deadline - 2.0:
        m3 = _local_search(best_mask, deadline)
        s3 = _popcount(m3)
        v3 = _violations_count(m3)
        if v3 < best_violations or (v3 == best_violations and s3 > best_size):
            best_mask = m3
            best_size = s3
            
    # Extract result
    result = [VALS[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
