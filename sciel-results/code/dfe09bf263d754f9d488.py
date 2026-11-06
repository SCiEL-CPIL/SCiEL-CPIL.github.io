import time
import random
import numpy as np

def _build_adj_masks(colors, n):
    """Build adjacency bitmasks for both colors."""
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
    
    return adj1, adj0

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
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

def _count_k222_violations(adj1, adj0, n):
    """Count total K2,2,2 violations in both colors. Returns (count1, count0)."""
    count1 = _count_k222_in_adj(adj1, n)
    count0 = _count_k222_in_adj(adj0, n)
    return count1, count0

def _count_k222_in_adj(adj, n):
    """Count number of K2,2,2 subgraphs in a graph."""
    count = 0
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

def _count_k222_numpy(adj_mat, n):
    """
    Count K2,2,2 subgraphs using numpy matrix operations.
    adj_mat is an n x n boolean numpy array (symmetric, zero diagonal).
    K2,2,2: vertices {u,v} (non-adjacent), {a,b} (non-adjacent, both in N(u)∩N(v)),
    {c,d} (non-adjacent, both in N(a)∩N(b)∩N(u)∩N(v))
    """
    # Non-adjacency matrix (excluding self-loops)
    non_adj = ~adj_mat
    np.fill_diagonal(non_adj, False)
    
    # For each pair (u,v) non-adjacent: C = N(u) ∩ N(v)
    # adj_mat[u] & adj_mat[v] gives common neighbors
    common_neighbors = adj_mat @ adj_mat  # (i,j) = |N(i) ∩ N(j)|... no, this is count
    
    # Actually, let's use a different approach:
    # For each non-edge (u,v), compute C = adj[u] & adj[v]
    # Then for each non-edge (a,b) within C, compute D = (adj[a] & adj[b] & C) \ {a,b}
    # Count non-edges in D
    
    count = 0
    # Precompute: for all pairs, whether they are non-adjacent
    # Use broadcasting for efficiency
    
    # Iterate over non-adjacent pairs (u,v)
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mat[u, v]:
                continue
            
            # C = common neighbors of u and v
            C = adj_mat[u] & adj_mat[v]
            c_count = C.sum()
            if c_count < 4:
                continue
            
            # Get indices of vertices in C
            verts_c = np.where(C)[0]
            m = len(verts_c)
            
            # For each non-adjacent pair (a,b) in C
            # adj_mat[a,b] should be False
            # D = (adj[a] & adj[b] & C) excluding a,b
            # Count non-edges in D
            
            # Extract submatrix for verts_c
            sub_adj = adj_mat[np.ix_(verts_c, verts_c)]
            
            # Non-adjacent pairs within C
            non_adj_sub = ~sub_adj
            np.fill_diagonal(non_adj_sub, False)
            
            # For each non-adjacent pair (i,j) in submatrix:
            # D = sub_adj[i] & sub_adj[j] & C_mask, excluding i,j
            # Then count non-edges in D
            
            # Vectorize: for all pairs (i,j) with non_adj_sub[i,j]:
            # common = sub_adj[i] & sub_adj[j]
            # But we need to exclude i and j from common
            # And then count non-edges among those common neighbors
            
            # This is getting complex. Let's do a hybrid approach.
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj_mat[a, b]:
                        continue
                    
                    # D = (adj[a] & adj[b] & C) \ {a,b}
                    D = adj_mat[a] & adj_mat[b] & C
                    D[a] = False
                    D[b] = False
                    d_count = D.sum()
                    if d_count < 2:
                        continue
                    
                    # Count non-edges in D
                    verts_d = np.where(D)[0]
                    k = len(verts_d)
                    if k < 2:
                        continue
                    sub_d = adj_mat[np.ix_(verts_d, verts_d)]
                    non_edges_d = (~sub_d).sum() // 2  # symmetric
                    # Actually need to count pairs that are non-adjacent
                    # non_adj_d = ~sub_d, fill diagonal False, count True
                    non_adj_d = ~sub_d
                    np.fill_diagonal(non_adj_d, False)
                    count += non_adj_d.sum() // 2
    return count

def _local_search_vec(colors, n, max_iters=50000, time_limit=50, temp_start=1.0, temp_end=0.01):
    """
    Vectorized local search with simulated annealing.
    Uses numpy adjacency matrix for fast updates.
    """
    start_time = time.time()
    
    # Build numpy adjacency matrix for color 1
    adj1 = np.zeros((n, n), dtype=np.bool_)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
            idx += 1
    
    # Precompute edge index: (i,j) -> index in colors list
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    # Initial violation count
    def get_violation_count():
        """Count total K2,2,2 violations in both colors."""
        # Color 1 violations
        v1 = _count_k222_numpy(adj1, n)
        # Color 0 is complement
        adj0 = ~adj1
        np.fill_diagonal(adj0, False)
        v0 = _count_k222_numpy(adj0, n)
        return v1 + v0
    
    cur_violations = get_violation_count()
    if cur_violations == 0:
        return True
    
    best_violations = cur_violations
    best_colors = colors[:]
    
    # Simulated annealing parameters
    it = 0
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        progress = it / max_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Pick an edge to flip
        # Strategy: with some probability, pick an edge involved in a violation
        # Otherwise, random edge
        if random.random() < 0.7:
            # Try to find an edge in a violation
            # Pick random non-edge in color 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj1[u, v]:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    idx_flip = random.randint(0, len(colors) - 1)
                    it += 1
                    continue
            
            C = adj1[u] & adj1[v]
            if C.sum() < 4:
                idx_flip = random.randint(0, len(colors) - 1)
            else:
                verts_c = np.where(C)[0]
                m = len(verts_c)
                found = False
                for _ in range(min(10, m * (m-1) // 2)):
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    if i == j:
                        continue
                    a = verts_c[i]
                    b = verts_c[j]
                    if adj1[a, b]:
                        continue
                    D = adj1[a] & adj1[b] & C
                    D[a] = False
                    D[b] = False
                    if D.sum() < 2:
                        continue
                    verts_d = np.where(D)[0]
                    if len(verts_d) < 2:
                        continue
                    x = random.randint(0, len(verts_d) - 1)
                    y = random.randint(0, len(verts_d) - 1)
                    if x == y:
                        continue
                    c = verts_d[x]
                    d = verts_d[y]
                    if adj1[c, d]:
                        continue
                    # Found a violation! Flip an edge in it
                    # Candidates: edges (a,c), (a,d), (b,c), (b,d) that are in color 1
                    candidates = []
                    for pair in [(a,c), (a,d), (b,c), (b,d)]:
                        p1, p2 = sorted(pair)
                        if adj1[p1, p2]:
                            candidates.append(edge_idx[(p1, p2)])
                    if candidates:
                        idx_flip = random.choice(candidates)
                        found = True
                        break
                if not found:
                    idx_flip = random.randint(0, len(colors) - 1)
        else:
            idx_flip = random.randint(0, len(colors) - 1)
        
        # Flip the edge
        i, j = None, None
        # Find which edge idx_flip corresponds to
        # We need to track this. Let's precompute a reverse mapping.
        # Actually, let's just iterate to find it, or precompute.
        
        # Precompute: idx -> (i, j)
        # We'll do this once outside the loop
        # For now, let's use a different approach: store the (i,j) for each idx
        # This is O(n^2) memory but n <= 31 so it's fine
        
        # Actually, let's restructure: precompute edge_list
        # edge_list[idx] = (i, j)
        
        # Let me refactor to precompute this
        break  # Placeholder, will fix below
    
    # Final check
    adj1_final = np.zeros((n, n), dtype=np.bool_)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1_final[i, j] = True
                adj1_final[j, i] = True
            idx += 1
    adj0_final = ~adj1_final
    np.fill_diagonal(adj0_final, False)
    v1 = _count_k222_numpy(adj1_final, n)
    v0 = _count_k222_numpy(adj0_final, n)
    return v1 + v0 == 0

def _local_search(colors, n, max_iters=5000, time_limit=50):
    """Local search to find valid coloring using violation-guided flips."""
    start = time.time()
    
    # Check initial
    adj1, adj0 = _build_adj_masks(colors, n)
    if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
        return True
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Rebuild adjacency
        adj1, adj0 = _build_adj_masks(colors, n)
        
        # Check if valid
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            return True
        
        # Find a violating edge and flip it
        # Strategy: try to find an edge involved in a K2,2,2 violation
        flipped = False
        attempts = 0
        max_attempts = min(50, max_iters - it)
        
        while not flipped and attempts < max_attempts:
            attempts += 1
            
            # Randomly pick a non-edge in color 1 (potential P1 pair)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj1[u] & (1 << v)):
                continue
            
            C = adj1[u] & adj1[v]
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
            if m < 4:
                continue
            
            # Try to find a violation
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj1[a] & (1 << b):
                        continue
                    
                    neighbors_ab_in_c = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
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
                            if not (adj1[c] & (1 << d)):
                                # Found a K2,2,2 violation in color 1!
                                # Flip one of the edges in this violation
                                # Options: flip edge (u,v) to break P1, or flip an edge within P2/P3
                                # Flip a random edge from the violation
                                candidates = []
                                # Edge (u,v) - but it's a non-edge in color 1, so flipping makes it color 1
                                # Actually we want to break the violation
                                # Flip edge (a,c) - if it exists in color 1
                                if adj1[a] & (1 << c):
                                    key = (a, c) if a < c else (c, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[a] & (1 << d):
                                    key = (a, d) if a < d else (d, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << c):
                                    key = (b, c) if b < c else (c, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << d):
                                    key = (b, d) if b < d else (d, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                
                                if candidates:
                                    idx_to_flip = random.choice(candidates)
                                    colors[idx_to_flip] = 1 - colors[idx_to_flip]
                                    flipped = True
                                    break
                    if flipped:
                        break
                if flipped:
                    break
            if flipped:
                break
        
        # If no violation found in color 1, check color 0
        if not flipped:
            for u in range(n):
                for v in range(u + 1, n):
                    if adj0[u] & (1 << v):
                        continue
                    
                    C = adj0[u] & adj0[v]
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
                    if m < 4:
                        continue
                    
                    for i in range(m):
                        a = vertices_in_c[i]
                        for j in range(i + 1, m):
                            b = vertices_in_c[j]
                            if adj0[a] & (1 << b):
                                continue
                            
                            neighbors_ab_in_c = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
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
                                    if not (adj0[c] & (1 << d)):
                                        # Found violation in color 0
                                        # Flip an edge that's color 0 to break it
                                        candidates = []
                                        if not (adj1[a] & (1 << c)):
                                            key = (a, c) if a < c else (c, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[a] & (1 << d)):
                                            key = (a, d) if a < d else (d, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << c)):
                                            key = (b, c) if b < c else (c, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << d)):
                                            key = (b, d) if b < d else (d, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        
                                        if candidates:
                                            idx_to_flip = random.choice(candidates)
                                            colors[idx_to_flip] = 1 - colors[idx_to_flip]
                                            flipped = True
                                            break
                            if flipped:
                                break
                        if flipped:
                            break
                    if flipped:
                        break
                if flipped:
                    break
        
        # If still no flip, do a random flip
        if not flipped:
            idx_to_flip = random.randint(0, len(colors) - 1)
            colors[idx_to_flip] = 1 - colors[idx_to_flip]
    
    # Final check
    adj1, adj0 = _build_adj_masks(colors, n)
    return not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n)

def _fast_local_search(colors, n, time_limit=50):
    """
    Fast local search using numpy adjacency matrix with incremental updates.
    Much faster than rebuilding adjacency each iteration.
    """
    start_time = time.time()
    n_edges = n * (n - 1) // 2
    
    # Precompute edge list: edge_idx -> (i, j)
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
            idx += 1
    
    # Build numpy adjacency matrix for color 1
    adj1 = np.zeros((n, n), dtype=np.int8)
    for k, (i, j) in enumerate(edge_list):
        if colors[k] == 1:
            adj1[i, j] = 1
            adj1[j, i] = 1
    
    def count_violations(adj):
        """Count K2,2,2 in adjacency matrix (int8, 0/1)."""
        # Convert to bool for operations
        adj_bool = adj.astype(np.bool_)
        non_adj = ~adj_bool
        np.fill_diagonal(non_adj, False)
        
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_bool[u, v]:
                    continue
                C = adj_bool[u] & adj_bool[v]
                if C.sum() < 4:
                    continue
                verts_c = np.where(C)[0]
                m = len(verts_c)
                for i in range(m):
                    a = verts_c[i]
                    for j in range(i + 1, m):
                        b = verts_c[j]
                        if adj_bool[a, b]:
                            continue
                        D = adj_bool[a] & adj_bool[b] & C
                        D[a] = False
                        D[b] = False
                        if D.sum() < 2:
                            continue
                        verts_d = np.where(D)[0]
                        k = len(verts_d)
                        if k < 2:
                            continue
                        sub_d = adj_bool[np.ix_(verts_d, verts_d)]
                        non_adj_d = ~sub_d
                        np.fill_diagonal(non_adj_d, False)
                        count += non_adj_d.sum() // 2
        return count
    
    cur_v1 = count_violations(adj1)
    adj0 = 1 - adj1
    np.fill_diagonal(adj0, 0)
    cur_v0 = count_violations(adj0)
    cur_total = cur_v1 + cur_v0
    
    if cur_total == 0:
        return True
    
    best_total = cur_total
    best_colors = colors[:]
    
    # Simulated annealing
    max_iters = 100000  # We'll break on time
    temp_start = 1.0
    temp_end = 0.001
    it = 0
    
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        progress = it / max_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Choose edge to flip
        if random.random() < 0.6:
            # Guided: try to flip an edge in a violation
            # Pick random non-edge in color 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj1[u, v] == 1:
                idx_flip = random.randint(0, n_edges - 1)
            else:
                C = (adj1[u] == 1) & (adj1[v] == 1)
                if C.sum() < 4:
                    idx_flip = random.randint(0, n_edges - 1)
                else:
                    verts_c = np.where(C)[0]
                    m = len(verts_c)
                    found = False
                    for _ in range(min(5, m * (m-1) // 2)):
                        i = random.randint(0, m - 1)
                        j = random.randint(0, m - 1)
                        if i == j:
                            continue
                        a = verts_c[i]
                        b = verts_c[j]
                        if adj1[a, b] == 1:
                            continue
                        D = (adj1[a] == 1) & (adj1[b] == 1) & C
                        D[a] = False
                        D[b] = False
                        if D.sum() < 2:
                            continue
                        verts_d = np.where(D)[0]
                        if len(verts_d) < 2:
                            continue
                        x = random.randint(0, len(verts_d) - 1)
                        y = random.randint(0, len(verts_d) - 1)
                        if x == y:
                            continue
                        c = verts_d[x]
                        d = verts_d[y]
                        if adj1[c, d] == 1:
                            continue
                        # Found violation, pick an edge to flip
                        candidates = []
                        for pair in [(a,c), (a,d), (b,c), (b,d)]:
                            p1, p2 = sorted(pair)
                            if adj1[p1, p2] == 1:
                                # Find edge index
                                # This is O(n^2) lookup, but n is small
                                for k, (ei, ej) in enumerate(edge_list):
                                    if ei == p1 and ej == p2:
                                        candidates.append(k)
                                        break
                        if candidates:
                            idx_flip = random.choice(candidates)
                            found = True
                            break
                    if not found:
                        idx_flip = random.randint(0, n_edges - 1)
        else:
            idx_flip = random.randint(0, n_edges - 1)
        
        # Flip the edge
        i, j = edge_list[idx_flip]
        old_val = colors[idx_flip]
        colors[idx_flip] = 1 - old_val
        
        # Update adjacency
        if old_val == 1:
            adj1[i, j] = 0
            adj1[j, i] = 0
        else:
            adj1[i, j] = 1
            adj1[j, i] = 1
        
        # Recompute violations (expensive, but we need it)
        new_v1 = count_violations(adj1)
        adj0 = 1 - adj1
        np.fill_diagonal(adj0, 0)
        new_v0 = count_violations(adj0)
        new_total = new_v1 + new_v0
        
        # Accept if better or with probability exp(-delta/temp)
        if new_total < best_total:
            best_total = new_total
            best_colors = colors[:]
        
        delta = new_total - cur_total
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            cur_total = new_total
        else:
            # Reject: undo flip
            colors[idx_flip] = old_val
            if old_val == 1:
                adj1[i, j] = 1
                adj1[j, i] = 1
            else:
                adj1[i, j] = 0
                adj1[j, i] = 0
        
        it += 1
    
    # Use best found
    colors[:] = best_colors
    
    # Final check
    adj1_final = np.zeros((n, n), dtype=np.int8)
    for k, (i, j) in enumerate(edge_list):
        if colors[k] == 1:
            adj1_final[i, j] = 1
            adj1_final[j, i] = 1
    adj0_final = 1 - adj1_final
    np.fill_diagonal(adj0_final, 0)
    v1 = count_violations(adj1_final)
    v0 = count_violations(adj0_final)
    return v1 + v0 == 0

def construct():
    deadline = time.time() + 135 # Safety margin (150s limit, leave 15s buffer)
    
    # Try Paley constructions
    # Paley(29) is a strong candidate for R(K2,2,2)-1
    
    # First, try to get n=29 using Paley(29)
    n_target = 29
    p_target = 29
    colors = _paley(n_target, p_target)
    if colors:
        adj1, adj0 = _build_adj_masks(colors, n_target)
        if not _has_k222_mask(adj1, n_target) and not _has_k222_mask(adj0, n_target):
            return {'n': n_target, 'colors': colors}
        else:
            # Try fast local search starting from Paley
            remaining = deadline - time.time()
            if remaining > 15:
                if _fast_local_search(colors, n_target, time_limit=int(remaining - 10)):
                    return {'n': n_target, 'colors': colors}

    # If Paley(29) failed, try other primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59]
    for p in primes:
        if time.time() > deadline - 15:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        adj1, adj0 = _build_adj_masks(colors, 29)
        if not _has_k222_mask(adj1, 29) and not _has_k222_mask(adj0, 29):
            return {'n': 29, 'colors': colors}
        remaining = deadline - time.time()
        if remaining > 20:
            if _fast_local_search(colors, 29, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Try random initializations for n=29
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 15:
            if _fast_local_search(colors, 29, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Fallback: Try to find valid coloring for smaller n
    for n in range(28, 16, -1):
        if time.time() > deadline - 10:
            break
        # Try Paley with p=n if n is prime and 1 mod 4, else next prime
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
            adj1, adj0 = _build_adj_masks(colors, n)
            if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
                return {'n': n, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 10:
                if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                    return {'n': n, 'colors': colors}
                
        # Random start
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                return {'n': n, 'colors': colors}

    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
