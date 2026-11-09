import time
import numpy as np

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    Tries Paley tournaments on primes p ≡ 3 (mod 4) from smallest to largest,
    returning the first one that satisfies the 4-dominator property.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Primes of form 4k+3, from 31 upward (lower bound is 31)
    primes_4k3 = [31, 43, 47, 59, 67, 79, 83]
    
    for p in primes_4k3:
        if time.time() > deadline:
            break
        if p < 5:
            continue
            
        # Compute quadratic residues mod p
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        # Build adjacency: adj[i][j] = True if i -> j
        # Use the Paley rule: i -> j iff (j-i) mod p is a quadratic residue
        # For i < j: diff = j - i; for i > j: diff = (j - i) % p = j - i + p
        adj = np.zeros((p, p), dtype=bool)
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if squares[diff]:
                    adj[i][j] = True
                else:
                    adj[j][i] = True
        
        # Build dominance masks: dom_mask[v] has bit u set if v -> u
        dom_mask = np.zeros(p, dtype=np.uint64)
        for v in range(p):
            mask = 0
            for u in range(p):
                if v != u and adj[v][u]:
                    mask |= (1 << u)
            dom_mask[v] = mask
        
        # Check: for every 4-subset S, exists v notin S with v->all in S
        # Equivalently: no 4-subset S is such that for all v notin S, v does not dominate all of S
        # i.e., for all v notin S, there exists u in S with not (v->u), i.e., u->v or v==u
        
        # Use itertools.combinations for clarity, but that's slow.
        # Use nested loops with early termination.
        
        valid = True
        p_local = p
        dm = dom_mask
        for a in range(p_local):
            if time.time() > deadline:
                valid = False
                break
            for b in range(a+1, p_local):
                for c in range(b+1, p_local):
                    for d in range(c+1, p_local):
                        # Check if any v not in {a,b,c,d} dominates all four
                        found = False
                        for v in range(p_local):
                            if v == a or v == b or v == c or v == d:
                                continue
                            m = dm[v]
                            if (m & (1 << a)) and (m & (1 << b)) and (m & (1 << c)) and (m & (1 << d)):
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                        if time.time() > deadline:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    if squares[diff]:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
    
    # Fallback: return the known valid Paley tournament on 67
    p = 67
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if squares[diff]:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
