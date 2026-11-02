"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # Try to find a 5-regular Cayley graph on PSL(2, 13) with girth >= 9.
    # If that fails, fall back to L-U-W.
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # Try PSL(2, 13) Cayley graph search
    result = _cayley_psl2_13_search(deadline)
    if result is not None:
        return result
        
    # Fallback: use L-U-W construction
    return _luw_construct(k, g)

def _cayley_psl2_13_search(deadline):
    """
    Search for a 5-regular Cayley graph on PSL(2, 13) with girth >= 9.
    """
    # Construct PSL(2, 13)
    # Elements are 2x2 matrices with det 1, modulo scalars.
    # Since we work in PSL, we can represent elements by matrices with det 1.
    # Two matrices are equivalent if they are scalar multiples.
    # In GF(13), scalars are non-zero.
    # We can fix the (0,0) entry to 1 if possible, or use a canonical form.
    
    p = 13
    # Generate all matrices in SL(2, p)
    # a b
    # c d
    # ad - bc = 1
    
    elems = []
    elem_to_idx = {}
    
    # To avoid duplicates in PSL, we identify M and -M.
    # We can just generate SL(2, p) and then quotient by {I, -I}.
    # Size of SL(2, p) is p(p^2-1) = 13*168 = 2184.
    # Size of PSL(2, p) is 1092.
    
    # Generate all (a,b,c,d) in GF(p)^4 such that ad-bc=1
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # ad - bc = 1 => ad = 1 + bc
                # If a == 0, then -bc = 1 => bc = -1 = p-1.
                # If a != 0, d = (1 + bc) * a^{-1}
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % p != (p - 1):
                        continue
                    d = 0
                    # Check ad - bc = 0 - b*c = -bc = 1 mod p?
                    # b*c = p-1 => -bc = 1. Yes.
                    m = ((a, b), (c, d))
                    # Canonical form: if m[0][0] != 0, normalize to 1.
                    # If m[0][0] == 0, normalize m[0][1] to 1?
                    # Let's just use the tuple and handle equivalence.
                    key = _canon_psl(m, p)
                    if key not in elem_to_idx:
                        elem_to_idx[key] = len(elems)
                        elems.append(m)
                else:
                    inv_a = pow(a, p - 2, p)
                    for c in range(p):
                        d = ((1 + b * c) * inv_a) % p
                        m = ((a, b), (c, d))
                        key = _canon_psl(m, p)
                        if key not in elem_to_idx:
                            elem_to_idx[key] = len(elems)
                            elems.append(m)
    
    n = len(elems)
    # n should be 1092
    
    # Precompute multiplication table?
    # n=1092. Table 1092x1092.
    # Might be slow to build.
    # Instead, compute neighbors on the fly?
    # We need to check girth. BFS requires neighbors.
    # Let's build the adjacency list for a given S.
    
    # Find a random S of size 5.
    # S must be symmetric: s in S => s^{-1} in S.
    # 1 not in S.
    # If s = s^{-1}, then s^2 = 1.
    
    # Identify involutions (s^2=1, s!=1).
    # In PSL(2, p), involutions are conjugate to [[0, -1], [1, 0]]?
    # Let's just check s*s = I.
    
    # To speed up, we can pick k/2 = 2 pairs of inverses, or 1 pair + 3 involutions?
    # Size 5 is odd. So we must have at least one involution (s=s^-1).
    # Let s1 be an involution.
    # Then pick 2 pairs (s2, s2^-1), (s3, s3^-1).
    # Total 5 elements.
    
    # Find an involution.
    involution = None
    for i, m in enumerate(elems):
        if i == 0: continue # Identity
        prod = _mat_mul(m, m, p)
        if _is_identity(prod, p):
            involution = m
            break
    
    if involution is None:
        return None
        
    inv_idx = elem_to_idx[_canon_psl(involution, p)]
    
    # Get list of other elements
    # We need to pick 4 more elements, forming 2 inverse pairs.
    # Total elements ~ 1092.
    
    # Random search
    rng = random.Random(42)
    
    # Precompute inverses?
    # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]].
    def get_inv_idx(m, p):
        a, b = m[0]
        c, d = m[1]
        inv = ((d, p-b), (p-c, a))
        key = _canon_psl(inv, p)
        return elem_to_idx.get(key, -1)
    
    # Build list of indices excluding 0 (identity) and inv_idx
    candidates = [i for i in range(n) if i != 0 and i != inv_idx]
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Pick 2 pairs
        # Pick s2
        s2_idx = rng.choice(candidates)
        s2_inv_idx = get_inv_idx(elems[s2_idx], p)
        if s2_inv_idx == -1 or s2_inv_idx == s2_idx:
            # s2 is involution?
            # If s2 is involution, it's its own inverse.
            # We need distinct elements in S.
            # If s2 is involution, we just add s2.
            # But we need 4 more.
            # Let's simplify: pick 4 distinct indices, ensure symmetry.
            
            # Easier: Pick 5 distinct indices from candidates, 
            # then check if S is symmetric.
            # Prob of symmetry is low.
            
            # Better: Pick s2, add s2 and s2^-1.
            # If s2 == s2^-1, add s2.
            
            pass
        
        # Refined selection:
        # 1. Add involution s1.
        S = {inv_idx}
        
        # Need 4 more.
        # Pick s2 from candidates.
        # Add s2 and s2^-1.
        # If s2 == s2^-1, add s2.
        
        # We need to ensure no duplicates.
        
        # Let's pick 2 random elements x, y from candidates.
        # S = {s1, x, x^-1, y, y^-1}
        # Ensure all distinct.
        
        x_idx = rng.choice(candidates)
        y_idx = rng.choice(candidates)
        
        x_inv = get_inv_idx(elems[x_idx], p)
        y_inv = get_inv_idx(elems[y_idx], p)
        
        S = {inv_idx, x_idx, x_inv, y_idx, y_inv}
        
        if len(S) < 5:
            continue # Duplicates found
            
        # Check if 0 in S
        if 0 in S:
            continue
            
        # Check girth
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u in range(n):
            for s_idx in S:
                v = _mat_mul_idx(u, s_idx, elems, p, elem_to_idx)
                adj[u].append(v)
                
        # BFS for girth
        girth = _check_girth(adj, n, 9)
        if girth >= 9:
            # Build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    return None

def _canon_psl(m, p):
    # m is ((a,b),(c,d))
    # In PSL, m ~ -m.
    # We can normalize such that the first non-zero element in row-major order is positive?
    # Or just return the tuple, and use a set of (m, -m) for lookup?
    # For speed, let's return the tuple, and in elem_to_idx, we store both m and -m pointing to same index.
    # But _canon is used to generate keys.
    # Let's define a canonical representative.
    # If a != 0, scale so a=1.
    # If a == 0 and b != 0, scale so b=1.
    # If a == 0 and b == 0, then c != 0, scale so c=1.
    
    a, b = m[0]
    c, d = m[1]
    
    if a != 0:
        inv_a = pow(a, p-2, p)
        return ((1, b*inv_a % p), (c*inv_a % p, d*inv_a % p))
    elif b != 0:
        inv_b = pow(b, p-2, p)
        return ((0, 1), (c*inv_b % p, d*inv_b % p))
    elif c != 0:
        inv_c = pow(c, p-2, p)
        return ((0, 0), (1, d*inv_c % p))
    else:
        # Identity
        return ((1, 0), (0, 1))

def _mat_mul(m1, m2, p):
    a1, b1 = m1[0]
    c1, d1 = m1[1]
    a2, b2 = m2[0]
    c2, d2 = m2[1]
    return (( (a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p ),
            ( (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p ))

def _is_identity(m, p):
    return m[0][0] == 1 and m[0][1] == 0 and m[1][0] == 0 and m[1][1] == 1

def _mat_mul_idx(u, s_idx, elems, p, elem_to_idx):
    # Multiply elems[u] * elems[s_idx]
    m = _mat_mul(elems[u], elems[s_idx], p)
    key = _canon_psl(m, p)
    return elem_to_idx.get(key, -1)

def _check_girth(adj, n, min_girth):
    # BFS from each vertex
    # Return true if no cycle < min_girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        q = deque()
        q.append(start)
        dist[start] = 0
        parent[start] = -2
        
        while q:
            u = q.popleft()
            if dist[u] >= min_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Back edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return cycle_len
    return min_girth

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
