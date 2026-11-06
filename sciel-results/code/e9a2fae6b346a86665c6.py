import time
import random
import numpy as np

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    """
    if n < 6:
        return False
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            verts_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                verts_in_c.append(v_idx)
                mask ^= lsb
            m = len(verts_in_c)
            if m < 4:
                continue
            for i in range(m):
                a = verts_in_c[i]
                for j in range(i + 1, m):
                    b = verts_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    if k < 2:
                        continue
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                return True
    return False

def _build_adj_from_colors(colors, n):
    """Build adjacency bitmasks from color list."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _is_valid_fast(colors, n):
    """Fast validity check using bitmask method."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj1 = _build_adj_from_colors(colors, n)
    if _has_k222_mask(adj1, n):
        return False
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    if _has_k222_mask(adj0, n):
        return False
    return True

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

def _find_violation_edge(adj, n):
    """Find an edge (i,j) that participates in a K2,2,2, return (i,j) or None."""
    if n < 6:
        return None
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            verts_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                verts_in_c.append(v_idx)
                mask ^= lsb
            m = len(verts_in_c)
            if m < 4:
                continue
            for i in range(m):
                a = verts_in_c[i]
                for j in range(i + 1, m):
                    b = verts_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                    k = len(verts_ab)
                    if k < 2:
                        continue
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                # Found K2,2,2: {u,v}, {a,b}, {c,d}
                                # Return a non-edge pair to flip
                                return (u, v)
    return None

def _local_search_smart(colors, n, max_iters=50000, time_limit=90):
    """
    Local search: prioritize flipping edges that are in many K2,2,2 violations.
    Uses bitmask adjacency for fast updates.
    """
    start = time.time()
    
    if _is_valid_fast(colors, n):
        return True
    
    edge_ij = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_ij.append((i, j))
    
    # Build adjacency bitmasks
    adj1 = _build_adj_from_colors(colors, n)
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    # Track which edges are in violations
    violation_count = np.zeros(len(colors), dtype=int)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Check current validity
        has_v1 = _has_k222_mask(adj1, n)
        has_v0 = _has_k222_mask(adj0, n)
        
        if not has_v1 and not has_v0:
            return True
        
        # Find a violation and flip an edge in it
        if has_v1:
            edge = _find_violation_edge(adj1, n)
            if edge is None:
                edge = _find_violation_edge(adj0, n)
        else:
            edge = _find_violation_edge(adj0, n)
        
        if edge is None:
            # Fallback: random edge
            if np.sum(violation_count) > 0:
                probs = violation_count / np.sum(violation_count)
                idx = np.random.choice(len(colors), p=probs)
            else:
                idx = random.randint(0, len(colors) - 1)
        else:
            # Find edge index
            i, j = edge
            # Compute edge index: sum of (n-1-k) for k < i, plus (j-1-i)
            idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
        i, j = edge_ij[idx]
        
        # Flip
        colors[idx] = 1 - colors[idx]
        if colors[idx] == 1:
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
        else:
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
        
        # Check validity
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            return True
    
    return _is_valid_fast(colors, n)

def _local_search_vec(colors, n, max_iters=100000, time_limit=90):
    """
    Fast local search using NumPy vectorized operations for adjacency updates
    and periodic validity checks.
    """
    start = time.time()
    
    if _is_valid_fast(colors, n):
        return True
    
    edge_ij = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_ij.append((i, j))
    
    # Build adjacency matrices
    adj1 = np.zeros((n, n), dtype=bool)
    adj0 = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
            else:
                adj0[i, j] = True
                adj0[j, i] = True
            idx += 1
    
    check_interval = max(1, max_iters // 50)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Pick random edge
        idx = random.randint(0, len(colors) - 1)
        i, j = edge_ij[idx]
        
        # Flip
        colors[idx] = 1 - colors[idx]
        if colors[idx] == 1:
            adj1[i, j] = True
            adj1[j, i] = True
            adj0[i, j] = False
            adj0[j, i] = False
        else:
            adj0[i, j] = True
            adj0[j, i] = True
            adj1[i, j] = False
            adj1[j, i] = False
        
        # Periodic check
        if (it + 1) % check_interval == 0:
            # Quick check using bitmask
            adj1_mask = [0] * n
            adj0_mask = [0] * n
            for u in range(n):
                m1 = 0
                m0 = 0
                for v in range(n):
                    if v == u:
                        continue
                    if adj1[u, v]:
                        m1 |= (1 << v)
                    if adj0[u, v]:
                        m0 |= (1 << v)
                adj1_mask[u] = m1
                adj0_mask[u] = m0
            
            if not _has_k222_mask(adj1_mask, n) and not _has_k222_mask(adj0_mask, n):
                return True
    
    # Final check
    return _is_valid_fast(colors, n)

def construct():
    deadline = time.time() + 135  # Safety margin (150s limit)
    
    # Try Paley constructions for n=29
    n_target = 29
    colors = _paley(n_target, n_target)
    if colors:
        if _is_valid_fast(colors, n_target):
            return {'n': n_target, 'colors': colors}
        # Local search from Paley start
        remaining = deadline - time.time()
        if remaining > 20:
            if _local_search_smart(colors, n_target, max_iters=30000, time_limit=min(80, remaining - 10)):
                return {'n': n_target, 'colors': colors}
    
    # Try other primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59, 61]
    for p in primes:
        if time.time() > deadline - 30:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        if _is_valid_fast(colors, 29):
            return {'n': 29, 'colors': colors}
        remaining = deadline - time.time()
        if remaining > 15:
            if _local_search_smart(colors, 29, max_iters=15000, time_limit=min(40, remaining - 10)):
                return {'n': 29, 'colors': colors}
    
    # Random restarts for n=29
    attempts = 0
    while time.time() < deadline - 40 and attempts < 15:
        attempts += 1
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 20:
            if _local_search_smart(colors, 29, max_iters=20000, time_limit=min(60, remaining - 15)):
                return {'n': 29, 'colors': colors}
    
    # Try n=28
    if time.time() < deadline - 30:
        colors = _paley(28, 29)
        if colors:
            if _is_valid_fast(colors, 28):
                return {'n': 28, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 28, max_iters=15000, time_limit=min(40, remaining - 10)):
                    return {'n': 28, 'colors': colors}
        
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            colors = [random.randint(0, 1) for _ in range(28 * 27 // 2)]
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 28, max_iters=10000, time_limit=min(30, remaining - 10)):
                    return {'n': 28, 'colors': colors}
    
    # Try n=27
    if time.time() < deadline - 25:
        colors = _paley(27, 29)
        if colors:
            if _is_valid_fast(colors, 27):
                return {'n': 27, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search_smart(colors, 27, max_iters=10000, time_limit=min(30, remaining - 10)):
                    return {'n': 27, 'colors': colors}
        
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            colors = [random.randint(0, 1) for _ in range(27 * 26 // 2)]
            remaining = deadline - time.time()
            if remaining > 10:
                if _local_search_smart(colors, 27, max_iters=8000, time_limit=min(20, remaining - 8)):
                    return {'n': 27, 'colors': colors}
    
    # Fallback: try smaller n with Paley
    for n in range(26, 15, -1):
        if time.time() > deadline - 10:
            break
        # Find suitable prime
        p = n
        while p < 50:
            is_prime = True
            if p < 2:
                is_prime = False
            else:
                for i in range(2, int(p ** 0.5) + 1):
                    if p % i == 0:
                        is_prime = False
                        break
            if is_prime and p % 4 == 1:
                break
            p += 1
        
        colors = _paley(n, p)
        if colors:
            if _is_valid_fast(colors, n):
                return {'n': n, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 10:
                if _local_search_smart(colors, n, max_iters=5000, time_limit=min(15, remaining - 8)):
                    return {'n': n, 'colors': colors}
        
        # Random
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _local_search_smart(colors, n, max_iters=5000, time_limit=min(15, remaining - 8)):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': []}
