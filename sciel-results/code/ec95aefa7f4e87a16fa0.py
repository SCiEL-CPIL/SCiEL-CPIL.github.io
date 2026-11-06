import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def elapsed():
        return time.time() - start_time

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return p, orientation

    def build_adj(orientation, n):
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orientation[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj

    def verify(tournament, n):
        adj = build_adj(tournament, n)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            found = False
            for u in range(n):
                if u in quad:
                    continue
                if adj[u, a] and adj[u, b] and adj[u, c] and adj[u, d]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Initial construction
    p_init, orientation = build_paley(67)
    n_orig = p_init
    adj_orig = build_adj(orientation, n_orig)

    active = list(range(n_orig))
    
    while elapsed() < DEADLINE:
        m = len(active)
        if m < 5:
            break
            
        # Build local adjacency
        adj_local = adj_orig[np.ix_(active, active)]
        
        # Precompute in_adj_mask: for each vertex x, bitmask of vertices u with u->x
        in_adj_mask = [0] * m
        for x in range(m):
            mask = 0
            for u in range(m):
                if adj_local[u, x]:
                    mask |= (1 << u)
            in_adj_mask[x] = mask
        
        # Find critical vertices: vertices that are the unique dominator of some 4-set
        critical = [False] * m
        
        quad_list = list(combinations(range(m), 4))
        
        for a, b, c, d in quad_list:
            if elapsed() > DEADLINE - 5:
                break
            dom_mask = in_adj_mask[a] & in_adj_mask[b] & in_adj_mask[c] & in_adj_mask[d]
            self_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dom_mask &= ~self_mask
            if dom_mask != 0:
                if dom_mask & (dom_mask - 1) == 0:
                    u = dom_mask.bit_length() - 1
                    critical[u] = True
        
        if all(critical):
            break
        
        # Remove non-critical vertices
        # Try to remove multiple at a time safely
        # For now, remove the first non-critical one
        for i in range(m):
            if not critical[i]:
                active.pop(i)
                break
        
    # Construct final orientation
    n_final = len(active)
    final_orientation = []
    for i in range(n_final):
        for j in range(i+1, n_final):
            orig_i = active[i]
            orig_j = active[j]
            if adj_orig[orig_i, orig_j]:
                final_orientation.append(0)
            else:
                final_orientation.append(1)
                
    # Final verification to be safe
    if not verify(final_orientation, n_final):
        return {'n': 67, 'orientation': orientation}

    return {'n': n_final, 'orientation': final_orientation}
