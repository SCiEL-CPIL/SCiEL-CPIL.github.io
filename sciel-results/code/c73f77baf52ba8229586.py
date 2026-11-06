import random
import time
import math

def construct():
    """
    Five-colourings of complete graphs without monochromatic C4s.
    Returns {'n': n, 'colors': [...]} where colors has one entry in {0,..,4} for each (i,j) with i<j.
    Goal: maximize n. Known best is 26 (Lazebnik-Woldar). 27 or 28 would be new.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # First, try to find n=26 coloring using a fast local search with good initialization
    result_26 = _find_coloring(26, time_limit=60.0, seed_time=start_time)
    
    if result_26 is not None:
        # Try to extend to 27
        if time.time() < deadline - 20:
            result_27 = _try_extend(result_26, 27, time_limit=45.0, seed_time=start_time)
            if result_27 is not None:
                return result_27
        return result_26
    
    # Fallback: try smaller n
    for n in [25, 24, 23, 22]:
        if time.time() > deadline - 10:
            break
        res = _find_coloring(n, time_limit=15.0, seed_time=start_time)
        if res is not None:
            return res
    
    # Absolute fallback: small valid coloring
    res = _find_coloring(10, time_limit=2.0, seed_time=start_time)
    if res is not None:
        return res
    
    return {'n': 1, 'colors': []}


def _find_coloring(n, time_limit=60.0, seed_time=None):
    """
    Find a valid 5-coloring of K_n with no monochromatic C4.
    Uses structured initialization + fast local search with delta evaluation.
    """
    if n <= 1:
        return {'n': n, 'colors': []}
    
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Structured initialization: use a cyclic pattern
    # For vertex i, edge (i, j) gets color (i + j) % 5
    colors = []
    for (i, j) in edges:
        colors.append((i + j) % 5)
    
    # Build adjacency bitmasks
    adj = _build_adj(n, colors, edges)
    
    violations = _count_violations(n, adj)
    
    if violations == 0:
        return {'n': n, 'colors': colors}
    
    # Fast local search with delta evaluation
    best_violations = violations
    best_colors = colors[:]
    
    start = time.time()
    temperature = 2.0
    cooling = 0.9997
    min_temp = 0.05
    
    while time.time() - start < time_limit:
        if violations == 0:
            return {'n': n, 'colors': colors}
        
        if violations < best_violations:
            best_violations = violations
            best_colors = colors[:]
        
        # Pick an edge involved in a violation if possible, else random
        e_idx = _pick_edge(n, adj, colors, edges, violations)
        i, j = edges[e_idx]
        old_c = colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        delta_remove = _delta_remove(n, adj, old_c, i, j)
        delta_add = _delta_add(n, adj, new_c, i, j)
        total_delta = delta_remove + delta_add
        
        if total_delta < 0 or (temperature > min_temp and random.random() < math.exp(-total_delta / temperature)):
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            colors[e_idx] = new_c
            violations += total_delta
        
        temperature *= cooling
    
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None


def _build_adj(n, colors, edges):
    """Build adjacency bitmasks for each color."""
    adj = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    return adj


def _count_violations(n, adj):
    """Count total C4 violations across all colors."""
    total = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                k = common.bit_count()
                if k >= 2:
                    total += k * (k - 1) // 2
    return total


def _delta_remove(n, adj, c, i, j):
    """
    Calculate change in violations when removing edge (i,j) from color c.
    Negative delta means violations decrease.
    """
    d = 0
    mask_i = adj[c][i]
    mask_j = adj[c][j]
    
    # For each v in N_c(j) \ {i}: j is a common neighbor of i and v
    temp = mask_j & ~(1 << i)
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        temp ^= lsb
        k_old = (mask_i & adj[c][v]).bit_count()
        d -= (k_old - 1)
    
    # For each v in N_c(i) \ {j}: i is a common neighbor of j and v
    temp = mask_i & ~(1 << j)
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        temp ^= lsb
        k_old = (mask_j & adj[c][v]).bit_count()
        d -= (k_old - 1)
    
    return d


def _delta_add(n, adj, c, i, j):
    """
    Calculate change in violations when adding edge (i,j) to color c.
    Positive delta means violations increase.
    """
    d = 0
    mask_i = adj[c][i]
    mask_j = adj[c][j]
    
    # For each v in N_c(j) \ {i}: j becomes a new common neighbor of i and v
    temp = mask_j & ~(1 << i)
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        temp ^= lsb
        k_old = (mask_i & adj[c][v]).bit_count()
        d += k_old
    
    # For each v in N_c(i) \ {j}: i becomes a new common neighbor of j and v
    temp = mask_i & ~(1 << j)
    while temp:
        lsb = temp & -temp
        v = lsb.bit_length() - 1
        temp ^= lsb
        k_old = (mask_j & adj[c][v]).bit_count()
        d += k_old
    
    return d


def _pick_edge(n, adj, colors, edges, violations):
    """
    Pick an edge to recolor. Prefer edges involved in violations.
    """
    if violations > 0 and random.random() < 0.6:
        # Try to find an edge in a violating C4
        for _ in range(20):
            c = random.randint(0, 4)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            common = adj[c][u] & adj[c][v]
            k = common.bit_count()
            if k >= 2:
                # Found a violation; pick one of the 4 edges in the C4
                comms = []
                temp = common
                while temp and len(comms) < 2:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    comms.append(w)
                    temp ^= lsb
                if len(comms) >= 2:
                    k1, k2 = comms[0], comms[1]
                    # The C4 is u-k1-v-k2-u
                    # Pick one of these edges
                    candidates = []
                    for a, b in [(u, k1), (k1, v), (v, k2), (k2, u)]:
                        if a > b:
                            a, b = b, a
                        # Find edge index
                        idx = a * n - a * (a + 1) // 2 + (b - a - 1)
                        if 0 <= idx < len(colors):
                            candidates.append(idx)
                    if candidates:
                        return random.choice(candidates)
    
    return random.randint(0, len(colors) - 1)


def _try_extend(result_26, target_n, time_limit=45.0, seed_time=None):
    """
    Try to extend a valid n=26 coloring to n=27.
    Add a new vertex and find colors for edges to it.
    """
    if result_26 is None:
        return None
    
    n_old = result_26['n']
    colors_old = result_26['colors']
    
    if n_old < target_n:
        # Need to add (target_n - n_old) vertices
        # For simplicity, just try n=27 from n=26
        pass
    
    num_edges_old = n_old * (n_old - 1) // 2
    num_edges_new = target_n * (target_n - 1) // 2
    
    # Start with old coloring, add new edges randomly
    colors = colors_old[:]
    new_edges = []
    for v in range(n_old, target_n):
        for u in range(v):
            new_edges.append((u, v))
    
    for (u, v) in new_edges:
        colors.append(random.randint(0, 4))
    
    edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
    adj = _build_adj(target_n, colors, edges)
    
    violations = _count_violations(target_n, adj)
    
    if violations == 0:
        return {'n': target_n, 'colors': colors}
    
    # Local search, but only modify new edges initially, then all edges
    start = time.time()
    temperature = 1.5
    cooling = 0.9996
    min_temp = 0.03
    
    best_violations = violations
    best_colors = colors[:]
    
    while time.time() - start < time_limit:
        if violations == 0:
            return {'n': target_n, 'colors': colors}
        
        if violations < best_violations:
            best_violations = violations
            best_colors = colors[:]
        
        # Prefer to modify new edges
        if random.random() < 0.7 and len(new_edges) > 0:
            # Pick a new edge
            idx = num_edges_old + random.randint(0, len(new_edges) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
        
        i, j = edges[idx]
        old_c = colors[idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        delta_remove = _delta_remove(target_n, adj, old_c, i, j)
        delta_add = _delta_add(target_n, adj, new_c, i, j)
        total_delta = delta_remove + delta_add
        
        if total_delta < 0 or (temperature > min_temp and random.random() < math.exp(-total_delta / temperature)):
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            colors[idx] = new_c
            violations += total_delta
        
        temperature *= cooling
    
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
    return None
