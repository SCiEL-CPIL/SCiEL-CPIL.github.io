"""
R_4(C6): 4-colouring of K_n with no monochromatic C6.
Different strategy: algebraic seed construction + fast incremental (delta) local search.
"""
import time
import random
import sys

DEADLINE_MARGIN = 5.0

def _edge_index(i, j, n):
    # lexicographic order: pairs (i,j) with i<j, grouped by i
    # index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n, num_colors=4):
    adj = [[0] * n for _ in range(num_colors)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _has_c6_bitset(adj, n):
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        mask_v5 = adj[v4] & adj[v0]
                        used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used
                        if mask_v5:
                            return True
    return False

def _is_valid(colors, n):
    adj = _build_adj(colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True

def _count_c6(adj, n):
    total = 0
    for c in range(4):
        a = adj[c]
        for v0 in range(n):
            m1 = a[v0]
            while m1:
                v1 = (m1 & -m1).bit_length() - 1
                m1 &= m1 - 1
                m2 = a[v1] & ~(1 << v0)
                while m2:
                    v2 = (m2 & -m2).bit_length() - 1
                    m2 &= m2 - 1
                    m3 = a[v2] & ~(1 << v0) & ~(1 << v1)
                    while m3:
                        v3 = (m3 & -m3).bit_length() - 1
                        m3 &= m3 - 1
                        m4 = a[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while m4:
                            v4 = (m4 & -m4).bit_length() - 1
                            m4 &= m4 - 1
                            m5 = a[v4] & a[v0]
                            used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            m5 &= ~used
                            if m5:
                                total += 1
    return total

def _algebraic_coloring(n):
    """
    Deterministic algebraic coloring using a prime field.
    color(i,j) = ( (i*j mod p) + (i-j)^2 mod p ) mod 4  -- not quite; let's use a known trick.
    
    Better: use a 4-coloring derived from a 1-factorization-like structure or from
    the quadratic residues of (i-j) mod p mapped to 4 classes.
    
    We'll use: for prime p >= n, color(i,j) = ( (i-j) * (i+j) mod p ) mod 4  -> not uniform.
    
    Simplest robust algebraic seed: use the "circle method" / cyclic coloring:
    color(i,j) = ((i - j) mod (2*n)) // (n//2)  -- messy.
    
    Let's use a finite field GF(p) with p prime, and define color based on the
    Legendre symbol of (i-j) mod p, giving 2 colors, then split by parity to get 4.
    Actually, a known good seed for Ramsey-type problems is:
    color(i,j) = ( (i * j) mod p ) % 4  for p large prime.
    This doesn't guarantee no C6, but it's a structured starting point.
    
    For our purposes, the algebraic part is just a *structured random* seed that
    is faster to generate and may have fewer C6s than pure random.
    """
    # Find a prime >= n
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        d = 3
        while d * d <= x:
            if x % d == 0:
                return False
            d += 2
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use a combination of field operations to spread colors
            val = (i * j) % p
            # Map to 4 colors using two bits of val
            colors[idx] = (val >> 1) % 4
            idx += 1
    return colors

def _incremental_flip_delta(adj, n, edge_idx, i, j, old_c, new_c):
    """
    Compute the change in total C6 count when flipping edge (i,j) from old_c to new_c.
    A 6-cycle is affected only if it uses edge (i,j) in the old color (removed)
    or in the new color (added).
    
    We count:
      - C6s in old_c that use edge (i,j): these are removed.
      - C6s in new_c that use edge (i,j): these are added.
    
    To count C6s using a specific edge (i,j) in color c:
      A C6 containing edge (i,j) in color c has the form:
      i - j - v2 - v3 - v4 - v5 - i  (all edges in color c)
      or equivalently any rotation.
      
      We can count by: for each v5 in N_c(i) \\ {j}, for each v4 in N_c(v5) \\ {i,j},
      for each v3 in N_c(v4) \\ {i,j,v5}, for each v2 in N_c(v3) \\ {i,j,v5,v4}
      such that v2 is in N_c(j) and v2 != v5.
      
      This is O(d^3) where d is degree.
    """
    def count_c6s_using_edge(a, i, j):
        """Count C6s in graph 'a' that use edge (i,j)."""
        cnt = 0
        a_i = a[i]
        a_j = a[j]
        # v5 in N(i) \\ {j}
        m_v5 = a_i & ~(1 << j)
        while m_v5:
            v5 = (m_v5 & -m_v5).bit_length() - 1
            m_v5 &= m_v5 - 1
            # v4 in N(v5) \\ {i, j}
            m_v4 = a[v5] & ~(1 << i) & ~(1 << j)
            while m_v4:
                v4 = (m_v4 & -m_v4).bit_length() - 1
                m_v4 &= m_v4 - 1
                # v3 in N(v4) \\ {i, j, v5}
                m_v3 = a[v4] & ~(1 << i) & ~(1 << j) & ~(1 << v5)
                while m_v3:
                    v3 = (m_v3 & -m_v3).bit_length() - 1
                    m_v3 &= m_v3 - 1
                    # v2 in N(v3) \\ {i, j, v5, v4} AND v2 in N(j) AND v2 != v5
                    m_v2 = a[v3] & a_j & ~(1 << i) & ~(1 << j) & ~(1 << v5) & ~(1 << v4)
                    cnt += bin(m_v2).count('1')
        return cnt
    
    removed = count_c6s_using_edge(adj[old_c], i, j)
    added = count_c6s_using_edge(adj[new_c], i, j)
    return added - removed

def _find_valid_coloring(n, time_limit=10.0, seed=42):
    random.seed(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    # Start with algebraic coloring
    colors = _algebraic_coloring(n)
    adj = _build_adj(colors, n)
    current_c6s = _count_c6(adj, n)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    # Precompute edge endpoints
    edge_endpoints = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_endpoints.append((i, j))
            idx += 1
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
        
        edge_idx = random.randint(0, num_edges - 1)
        i, j = edge_endpoints[edge_idx]
        old_c = colors[edge_idx]
        new_c = (old_c + 1 + random.randint(0, 2)) % 4
        
        delta = _incremental_flip_delta(adj, n, edge_idx, i, j, old_c, new_c)
        new_c6s = current_c6s + delta
        
        if new_c6s <= best_c6s:
            # Apply flip
            colors[edge_idx] = new_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            current_c6s = new_c6s
            best_c6s = new_c6s
            best_colors = colors[:]
        # else: don't apply (keep old)
    
    if best_c6s == 0:
        return best_colors
    return None

def _construct_known_17():
    # Try n=19, 18, 17, 16, ... with increasing time budgets
    attempts = [
        (19, 20.0),
        (18, 25.0),
        (17, 30.0),
        (16, 20.0),
        (15, 15.0),
        (14, 10.0),
        (13, 8.0),
        (12, 6.0),
        (11, 5.0),
        (10, 4.0),
        (9, 3.0),
        (8, 2.0),
        (7, 2.0),
        (6, 1.0),
        (5, 1.0),
    ]
    
    start = time.time()
    remaining = 150.0 - DEADLINE_MARGIN
    
    for n, tl in attempts:
        if time.time() - start > remaining - 5:
            break
        actual_tl = min(tl, remaining - 5)
        if actual_tl < 1.0:
            break
        result = _find_valid_coloring(n, time_limit=actual_tl)
        if result is not None:
            return {'n': n, 'colors': result}
        remaining = 150.0 - DEADLINE_MARGIN - (time.time() - start)
    
    # Fallback: n=4
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}

def construct():
    start_time = time.time()
    result = _construct_known_17()
    return {'n': result['n'], 'colors': result['colors']}
