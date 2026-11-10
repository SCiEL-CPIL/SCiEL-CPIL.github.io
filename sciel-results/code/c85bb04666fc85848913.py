import time
import random
import numpy as np
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For k=5, g=9, we attempt to find a cyclic Cayley graph.
    # If not found, fallback to LUW with corrected d.
    
    if k == 5 and g == 9:
        # Try to find a cyclic Cayley graph with n < 1250
        # Known best is 1116.
        
        best_n = 1250 # Default to LUW size
        best_edges = None
        
        # We will try to find a valid cyclic graph
        # Iterate n from 1116 to 1249
        # Step 2 to keep n even
        
        # To save time, we try a subset or limited trials.
        # Given the difficulty, we might just skip this if it's too slow.
        # But let's try a few.
        
        for n in range(1116, 1250, 2):
            if time.time() > deadline - 10: # Leave 10s for fallback
                break
                
            trials = 2 # Reduce trials to save time
            for _ in range(trials):
                if time.time() > deadline - 10:
                    break
                    
                low = n // 4
                high = 3 * n // 4
                
                valid_range = list(range(low, high))
                if n // 2 in valid_range:
                    valid_range.remove(n // 2)
                    
                if len(valid_range) < 2:
                    continue
                    
                s1, s2 = random.sample(valid_range, 2)
                
                if s2 == (-s1) % n:
                    continue
                    
                S = [n // 2, s1, (-s1) % n, s2, (-s2) % n]
                if len(set(S)) != 5:
                    continue
                
                # Check girth
                if _check_girth_numpy(n, S, g):
                    edges = []
                    for u in range(n):
                        for s in S:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
        # If no cyclic graph found, return LUW with corrected d
        return _luw_construct(k, g)
        
    # Fallback for other k, g
    return _luw_construct(k, g)

def _check_girth_numpy(n, S, g):
    S_arr = np.array(S)
    
    for L in range(3, g):
        seqs = list(product(S_arr, repeat=L))
        if not seqs:
            continue
            
        A = np.array(seqs)
        sums = A.sum(axis=1) % n
        zero_indices = np.where(sums == 0)[0]
        
        if len(zero_indices) == 0:
            continue
            
        A_zero = A[zero_indices]
        partial_sums = np.cumsum(A_zero, axis=1) % n
        
        for i in range(len(zero_indices)):
            ps = partial_sums[i]
            if (ps[:-1] == 0).any():
                return False
            if len(set(ps.tolist())) < L:
                return False
                
    return True

def _luw_construct(k, g):
    q = k
    # Corrected d calculation for LUW graphs
    # Girth of G(q,d) is 2d+1
    # We need 2d+1 >= g => d >= (g-1)/2
    d = (g + 1) // 2
    if d < 3: d = 3 # Ensure minimum d for validity if needed, though g>=5 usually
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
