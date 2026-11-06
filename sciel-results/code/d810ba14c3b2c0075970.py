import time
import random
import itertools
import numpy as np

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    Parts P1, P2, P3.
    No edges within parts.
    All edges between parts.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            vertices_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                vertices_in_c.append(v_idx)
                mask ^= lsb
            m = len(vertices_in_c)
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if neighbors_ab_in_c.bit_count() < 2:
                        continue
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                return True
    return False

def _count_violations_bitmask(colors, n):
    """Count total K2,2,2 violations in both color classes using bitmasks."""
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    count = 0
    # Count violations in color 1
    for u in range(n):
        for v in range(u + 1, n):
            if adj1[u] & (1 << v):
                continue
            C = adj1[u] & adj1[v]
            if C.bit_count() < 4:
                continue
            # Count pairs (a,b) in C that are non-edges, and for each count
            # pairs (c,d) in N(a,b)∩C\{a,b} that are non-edges
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts_c)
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj1[a] & (1 << b):
                        continue
                    nbr_ab = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
                    if nbr_ab.bit_count() < 2:
                        continue
                    verts_ab = []
                    mask2 = nbr_ab
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_ab.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj1[c] & (1 << d)):
                                count += 1
    # Count violations in color 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj0[u] & (1 << v):
                continue
            C = adj0[u] & adj0[v]
            if C.bit_count() < 4:
                continue
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts_c)
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj0[a] & (1 << b):
                        continue
                    nbr_ab = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
                    if nbr_ab.bit_count() < 2:
                        continue
                    verts_ab = []
                    mask2 = nbr_ab
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_ab.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj0[c] & (1 << d)):
                                count += 1
    return count

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _sa_search(colors, n, time_limit=60):
    """Simulated annealing to minimize K2,2,2 violations."""
    start = time.time()
    colors = list(colors)
    m = len(colors)
    
    # Precompute edge index mapping
    edge_to_vertices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_vertices.append((i, j))
            idx += 1
    
    def get_violations(cols):
        return _count_violations_bitmask(cols, n)
    
    current_viol = get_violations(colors)
    if current_viol == 0:
        return colors, True
    
    best_colors = list(colors)
    best_viol = current_viol
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start < time_limit:
        if T <= T_min:
            T = 10.0
        
        # Pick random edge to flip
        idx = random.randint(0, m - 1)
        old_color = colors[idx]
        colors[idx] = 1 - old_color
        
        new_viol = get_violations(colors)
        delta = new_viol - current_viol
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = list(colors)
                if best_viol == 0:
                    return best_colors, True
        else:
            colors[idx] = old_color
        
        T *= cooling
    
    return best_colors, (best_viol == 0)

def construct():
    deadline = time.time() + 140
    
    # Try Paley(29) first - known strong candidate
    n_target = 29
    p_target = 29
    colors = _paley(n_target, p_target)
    if colors:
        viol = _count_violations_bitmask(colors, n_target)
        if viol == 0:
            return {'n': n_target, 'colors': colors}
        # SA from Paley
        if time.time() < deadline - 10:
            best_colors, success = _sa_search(colors, n_target, time_limit=min(50, int(deadline - time.time()) - 10))
            if success:
                return {'n': n_target, 'colors': best_colors}
    
    # Try other primes for Paley
    primes = [31, 37, 41, 43, 47, 53, 59, 61]
    for p in primes:
        if time.time() > deadline - 10:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        viol = _count_violations_bitmask(colors, 29)
        if viol == 0:
            return {'n': 29, 'colors': colors}
        if time.time() < deadline - 10:
            best_colors, success = _sa_search(colors, 29, time_limit=min(20, int(deadline - time.time()) - 10))
            if success:
                return {'n': 29, 'colors': best_colors}
    
    # Random starts with SA
    for attempt in range(15):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        best_colors, success = _sa_search(colors, 29, time_limit=min(8, int(deadline - time.time()) - 10))
        if success:
            return {'n': 29, 'colors': best_colors}
    
    # Fallback: try smaller n
    for n in range(28, 16, -1):
        if time.time() > deadline - 5:
            break
        p = n
        while p < 31:
            is_prime = True
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime and p % 4 == 1:
                break
            p += 1
        colors = _paley(n, p)
        if colors:
            viol = _count_violations_bitmask(colors, n)
            if viol == 0:
                return {'n': n, 'colors': colors}
            if time.time() < deadline - 5:
                best_colors, success = _sa_search(colors, n, time_limit=min(5, int(deadline - time.time()) - 5))
                if success:
                    return {'n': n, 'colors': best_colors}
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        if time.time() < deadline - 5:
            best_colors, success = _sa_search(colors, n, time_limit=min(5, int(deadline - time.time()) - 5))
            if success:
                return {'n': n, 'colors': best_colors}
    
    return {'n': 1, 'colors': []}
