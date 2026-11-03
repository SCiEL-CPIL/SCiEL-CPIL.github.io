import time
import math
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _construct_lps(p):
    """
    Construct the Lubotzky-Phillips-Sarnak graph for prime p.
    Requires p % 5 == 1.
    Returns (n, edges).
    """
    if p % 5 != 1:
        raise ValueError("p must be 1 mod 5")
    
    # The graph is defined on PGL(2, p) or similar.
    # Standard construction:
    # Vertices are elements of PSL(2, p) / {I, -I} ?
    # Actually, the LPS graph X(p) is a Cayley graph of PSL(2, p) with generators.
    # For p=11, |PSL(2, 11)| = 660.
    # The graph is 5-regular.
    # Vertices: 0 .. 659.
    
    # We need the generators.
    # For p=11, the generators are matrices in PSL(2, 11).
    # A = [[0, 1], [-1, 0]]
    # B = [[1, 1], [0, 1]] ? No.
    
    # Let's use the standard explicit construction for p=11.
    # From literature:
    # The 5-regular graph on 1320 vertices?
    # Wait, |PSL(2, 11)| = 660.
    # The LPS graph is on |PSL(2, p)| vertices?
    # Yes, for p=11, n=660?
    # Let's check the size.
    # The prompt says LPS(11) has n=1320?
    # Actually, the LPS graph X(p) has p(p^2-1) vertices?
    # No, that's for PGL(2, p).
    # PSL(2, p) has p(p^2-1)/2 vertices.
    # For p=11, 11*120/2 = 660.
    # Is it 5-regular?
    # The degree is 5.
    # So n=660.
    # Girth of LPS(11) is 8.
    # If n=660, it's smaller than 1116? No, 660 < 1116.
    # But girth is 8. We need 9.
    # If we can break 8-cycles in the 660-vertex graph, we might get girth 9 or 10.
    # This is even better!
    
    # Let's construct the Cayley graph of PSL(2, 11).
    # We need to represent elements of PSL(2, 11).
    # Elements are 2x2 matrices with determinant 1, modulo scalar.
    # Since p=11, -1 is not a square? 11 is 3 mod 4.
    # So we can represent elements as (a, b, c, d) with ad-bc=1.
    
    # Generators for p=11:
    # S = [[0, 1], [-1, 0]]
    # T = [[1, 1], [0, 1]]
    # U = [[1, 0], [1, 1]] ?
    # The standard generating set for LPS is:
    # A = [[0, 1], [-1, 0]]
    # B = [[1, 1], [0, 1]]
    # C = [[1, 0], [1, 1]]
    # D = [[1, 1], [1, 0]] ?
    # Actually, the 5 generators are:
    # S, T, T^{-1}, U, U^{-1} ?
    
    # Let's use a known explicit edge list for LPS(11) if possible, or generate it.
    # Generating PSL(2, 11) elements:
    
    def mat_mul(A, B):
        return np.array([
            [ (A[0,0]*B[0,0] + A[0,1]*B[1,0]) % p, (A[0,0]*B[0,1] + A[0,1]*B[1,1]) % p ],
            [ (A[1,0]*B[0,0] + A[1,1]*B[1,0]) % p, (A[1,0]*B[0,1] + A[1,1]*B[1,1]) % p ]
        ])
    
    def mat_inv(A):
        det = (A[0,0]*A[1,1] - A[0,1]*A[1,0]) % p
        det_inv = pow(det, p-2, p)
        return np.array([
            [ A[1,1]*det_inv % p, -A[0,1]*det_inv % p ],
            [ -A[1,0]*det_inv % p, A[0,0]*det_inv % p ]
        ])
    
    # Generate all elements of PSL(2, p)
    # a, d can be anything, b, c determined by ad-bc=1.
    # If a=0, then -bc=1 => c = -b^{-1}.
    # If a!=0, then d = (1+bc)/a.
    
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    if b == 0: continue
                    if (-b * c) % p != 1: continue
                    d = 0
                else:
                    # (1+bc) must be divisible by a
                    num = (1 + b * c) % p
                    if num % a != 0: continue
                    d = (num // a) % p
                    # Check det
                    if (a * d - b * c) % p != 1: continue
                # Add element
                # Normalize? In PSL, M and -M are same.
                # For p=11, -1 = 10.
                # We can just use the matrix as is, and identify M ~ -M.
                # But for Cayley graph, we use group elements.
                # We need a unique representative.
                # Let's store (a, b, c, d).
                elements.append((a, b, c, d))
                
    # Remove duplicates (M and -M)
    unique_elements = []
    seen = set()
    for e in elements:
        # Check if -e is already in seen
        neg_e = ((p-e[0])%p, (p-e[1])%p, (p-e[2])%p, (p-e[3])%p)
        if e not in seen and neg_e not in seen:
            unique_elements.append(e)
            seen.add(e)
            seen.add(neg_e)
            
    n = len(unique_elements)
    # Map element to index
    elem_to_idx = {e: i for i, e in enumerate(unique_elements)}
    
    # Define generators
    # S = [[0, 1], [-1, 0]] -> (0, 1, 10, 0)
    # T = [[1, 1], [0, 1]] -> (1, 1, 0, 1)
    # T_inv = [[1, -1], [0, 1]] -> (1, 10, 0, 1)
    # U = [[1, 0], [1, 1]] -> (1, 0, 1, 1)
    # U_inv = [[1, 0], [-1, 1]] -> (1, 0, 10, 1)
    
    generators = [
        (0, 1, 10, 0),
        (1, 1, 0, 1),
        (1, 10, 0, 1),
        (1, 0, 1, 1),
        (1, 0, 10, 1)
    ]
    
    # Verify generators are in the group
    # They should be.
    
    edges = []
    for i, e in enumerate(unique_elements):
        for g in generators:
            # Multiply e * g
            # e is (a,b,c,d), g is (g0,g1,g2,g3)
            # M = [[a,b],[c,d]]
            # G = [[g0,g1],[g2,g3]]
            # P = M * G
            a, b, c, d = e
            g0, g1, g2, g3 = g
            p00 = (a*g0 + b*g2) % p
            p01 = (a*g1 + b*g3) % p
            p10 = (c*g0 + d*g2) % p
            p11 = (c*g1 + d*g3) % p
            prod = (p00, p01, p10, p11)
            
            # Find index of prod or -prod
            if prod in elem_to_idx:
                j = elem_to_idx[prod]
            else:
                neg_prod = ((p-p00)%p, (p-p01)%p, (p-p10)%p, (p-p11)%p)
                if neg_prod in elem_to_idx:
                    j = elem_to_idx[neg_prod]
                else:
                    # Should not happen
                    continue
            
            if i < j:
                edges.append((i, j))
                
    return n, edges

def _luw_construct(k, g):
    """Fallback LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj

def _girth_check_fast(adj, g):
    n = len(adj)
    r = (g - 1) // 2
    for v in range(n):
        depth = [-1] * n
        parent = [-1] * n
        depth[v] = 0
        parent[v] = -2
        frontier = [v]
        for d in range(1, r + 1):
            new_frontier = []
            for u in frontier:
                for w in adj[u]:
                    if depth[w] == -1:
                        depth[w] = d
                        parent[w] = u
                        new_frontier.append(w)
                    elif w != parent[u] and depth[w] < d:
                        cycle_len = depth[w] + 1 + d
                        if cycle_len < g:
                            return False
            frontier = new_frontier
            if not frontier:
                break
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Try LPS(11) + Local Search
    if k == 5 and g == 9:
        try:
            n_lps, edges_lps = _construct_lps(11)
            adj_lps = _build_adj(n_lps, edges_lps)
            
            # Check if LPS(11) already has girth >= 9
            if _girth_check_fast(adj_lps, g):
                return n_lps, edges_lps
            
            # Local Search to improve girth
            # Objective: Minimize short cycles
            # We will try to swap edges to break 8-cycles.
            # This is a heuristic.
            # Given time constraints, we might just return LPS if we can't improve quickly.
            # But LPS(11) has girth 8, so it's invalid.
            # We MUST return a valid graph.
            # If we don't improve it to girth 9, we fall back to LUW.
            
            # Try a few seconds of local search
            # ... (Implementation of local search would go here)
            # For now, if it's not valid, we skip.
            
        except Exception:
            pass
            
    # 2. Fallback to LUW
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
