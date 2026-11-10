# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    Uses degree-based vertex ordering for better pruning.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute: for each pair (i,j), the set of common neighbors
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
    
    # Precompute degree in the full graph for initial ordering
    full_deg = [adj_bitmasks[i].bit_count() for i in range(n)]
    
    def dfs(available, current_mask, current_size):
        nonlocal best
        # Pruning
        avail_count = available.bit_count()
        if current_size + avail_count <= best:
            return
            
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Pick vertex with highest degree in the available subgraph
        # This gives better pruning since high-degree vertices create more conflicts
        best_v = -1
        best_deg = -1
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj_bitmasks[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp -= v_bit
        
        v = best_v
        v_bit = 1 << v
        
        # Option 1: Include v
        # Check if adding v to current_mask creates a triangle
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            temp = N_v_in_cur
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                if common[v][u]:
                    safe_to_add = False
                    break
                temp -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            # Determine forbidden vertices: neighbors of v that are also neighbors of some vertex in current_mask
            # A vertex w in available is forbidden if w is adjacent to v AND w is adjacent to some u in current_mask
            # This would create triangle (v, w, u)
            forbidden = 0
            N_v_in_avail = adj_bitmasks[v] & available
            temp_w = N_v_in_avail
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                if common[v][w] & current_mask:
                    forbidden |= w_bit
                temp_w -= w_bit
            
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
            
        # Option 2: Exclude v
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _find_max_tf_set(adj_bitmasks, n):
    """Find the actual largest triangle-free set (not just size)."""
    best_size = 0
    best_set = 0
    full_mask = (1 << n) - 1
    
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
    
    def dfs(available, current_mask, current_size):
        nonlocal best_size, best_set
        if current_size + available.bit_count() <= best_size:
            return
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
        
        # Pick vertex with highest degree in available subgraph
        best_v = -1
        best_deg = -1
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj_bitmasks[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp -= v_bit
        
        v = best_v
        v_bit = 1 << v
        
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            temp = N_v_in_cur
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                if common[v][u]:
                    safe_to_add = False
                    break
                temp -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            forbidden = 0
            N_v_in_avail = adj_bitmasks[v] & available
            temp_w = N_v_in_avail
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                if common[v][w] & current_mask:
                    forbidden |= w_bit
                temp_w -= w_bit
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
        
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best_set, best_size

def _can_add_edge(bm, i, j, n):
    """Check if adding edge (i,j) keeps graph K4-free."""
    common_ij = bm[i] & bm[j]
    temp = common_ij
    while temp:
        k_bit = temp & -temp
        k = k_bit.bit_length() - 1
        if bm[i] & bm[j] & bm[k]:
            return False
        temp -= k_bit
    return True

def _targeted_local_search(bm, n, max_iter=300, seed=0, deadline=None):
    """
    Targeted local search that specifically tries to break up the largest 
    triangle-free set by adding edges within it.
    Uses systematic edge addition within the TF set, prioritized by impact.
    """
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    no_improve = 0
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Find the largest triangle-free set
        tf_mask, tf_size = _find_max_tf_set(bm, n)
        
        # Get vertices in the TF set
        tf_vertices = []
        temp = tf_mask
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            tf_vertices.append(v)
            temp -= v_bit
        
        if len(tf_vertices) < 2:
            break
        
        # Try to add ALL missing edges within the TF set
        # Prioritize by number of common neighbors (higher = more K4 risk, but also more impact)
        pairs = []
        for i in range(len(tf_vertices)):
            for j in range(i+1, len(tf_vertices)):
                a, b = tf_vertices[i], tf_vertices[j]
                if not (bm[a] & (1 << b)):
                    # Score this edge by how many TF vertices it's adjacent to
                    adj_count = 0
                    for k in tf_vertices:
                        if k != a and k != b:
                            if (bm[a] & (1 << k)) or (bm[b] & (1 << k)):
                                adj_count += 1
                    pairs.append((adj_count, a, b))
        
        # Sort by impact (descending)
        pairs.sort(key=lambda x: -x[0])
        
        improved = False
        for adj_count, i, j in pairs:
            if deadline and time.time() > deadline:
                break
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                    no_improve = 0
                    improved = True
                    # Re-find TF set after improvement
                    break
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        
        if not improved:
            # Try random edge additions/removals
            for _ in range(5):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                i, j = min(i, j), max(i, j)
                
                if not (bm[i] & (1 << j)):
                    if _can_add_edge(bm, i, j, n):
                        bm[i] |= (1 << j)
                        bm[j] |= (1 << i)
                        new_score = _max_tf_size_fast(bm, n)
                        if new_score < best_score:
                            best_score = new_score
                            best_bm = bm[:]
                            no_improve = 0
                            improved = True
                        else:
                            bm[i] ^= (1 << j)
                            bm[j] ^= (1 << i)
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
                    new_score = _max_tf_size_fast(bm, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_bm = bm[:]
                        no_improve = 0
                        improved = True
                    else:
                        bm[i] ^= (1 << j)
                        bm[j] ^= (1 << i)
        
        if not improved:
            no_improve += 1
            if no_improve > 20:
                break
        
        # Sync bm with best if we improved
        if improved:
            bm = best_bm[:]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _local_search_bitmask(bm, n, max_iter=500, seed=0, deadline=None):
    """Local search on bitmask representation."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    no_improve = 0
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        improved = False
        if not (bm[i] & (1 << j)):
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                    no_improve = 0
                    improved = True
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        else:
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
                no_improve = 0
                improved = True
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                break
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _two_opt_search(bm, n, max_iter=200, seed=0, deadline=None):
    """2-opt style: try swapping two edge states simultaneously."""
    rng = random.Random(seed)
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    no_improve = 0
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i1 = rng.randint(0, n - 1)
        j1 = rng.randint(0, n - 1)
        i2 = rng.randint(0, n - 1)
        j2 = rng.randint(0, n - 1)
        if i1 == j1 or i2 == j2:
            continue
        i1, j1 = min(i1, j1), max(i1, j1)
        i2, j2 = min(i2, j2), max(i2, j2)
        if (i1, j1) == (i2, j2):
            continue
        
        e1_exists = bool(bm[i1] & (1 << j1))
        e2_exists = bool(bm[i2] & (1 << j2))
        
        # Try flipping both
        bm[i1] ^= (1 << j1)
        bm[j1] ^= (1 << i1)
        bm[i2] ^= (1 << j2)
        bm[j2] ^= (1 << i2)
        
        # Check K4-free
        valid = True
        for a in range(n):
            for b in range(a + 1, n):
                if bm[a] & (1 << b):
                    common_ab = bm[a] & bm[b]
                    temp = common_ab
                    while temp:
                        c_bit = temp & -temp
                        c = c_bit.bit_length() - 1
                        if bm[a] & bm[b] & bm[c]:
                            valid = False
                            break
                        temp -= c_bit
                    if not valid:
                        break
            if not valid:
                break
        
        if valid:
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
                no_improve = 0
            else:
                bm[i1] ^= (1 << j1)
                bm[j1] ^= (1 << i1)
                bm[i2] ^= (1 << j2)
                bm[j2] ^= (1 << i2)
                no_improve += 1
        else:
            bm[i1] ^= (1 << j1)
            bm[j1] ^= (1 << i1)
            bm[i2] ^= (1 << j2)
            bm[j2] ^= (1 << i2)
            no_improve += 1
        
        if no_improve > 100:
            break
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _random_k4_free_graph(n, seed):
    """Generate a random K4-free graph using bitmask representation."""
    rng = random.Random(seed)
    bm = [0] * n
    
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for i, j in edges_to_try:
        if rng.random() < 0.5:
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def _perturbed_turan(n, seed, num_perturbations=5):
    """Generate a perturbed Turan graph by randomly adding/removing edges."""
    rng = random.Random(seed)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    bm[a] |= (1 << b)
                    bm[b] |= (1 << a)
    
    # Apply perturbations
    for _ in range(num_perturbations):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        if not (bm[i] & (1 << j)):
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
        else:
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    n = 18
    
    # 1. Baseline: Turan graph T(18, 3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_bm[a] |= (1 << b)
                    baseline_bm[b] |= (1 << a)
    
    baseline_score = _max_tf_size_fast(baseline_bm, n)
    best_score = baseline_score
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if baseline_bm[i] & (1 << j):
                best_edges.append([i, j])
    
    # 2. Perturbed Turan graphs with targeted local search
    for seed in range(60):
        if time.time() > deadline - 20:
            break
        
        bm, edges = _perturbed_turan(n, seed, num_perturbations=3)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        if time.time() > deadline - 25:
            break
        
        ls_edges, ls_score = _targeted_local_search(bm, n, max_iter=200, seed=seed, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # 3. Random restarts with targeted local search
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        
        bm, edges = _random_k4_free_graph(n, seed)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        if time.time() > deadline - 35:
            break
        
        ls_edges, ls_score = _targeted_local_search(bm, n, max_iter=150, seed=seed + 1000, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # 4. Final polish: 2-opt on best candidate
    if time.time() < deadline - 10:
        best_bm = [0] * n
        for e in best_edges:
            u, v = e
            best_bm[u] |= (1 << v)
            best_bm[v] |= (1 << u)
        
        polish_edges, polish_score = _two_opt_search(best_bm, n, max_iter=400, seed=999, deadline=deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_edges = polish_edges
        
        # One more targeted pass
        if time.time() < deadline - 5:
            final_bm = [0] * n
            for e in best_edges:
                u, v = e
                final_bm[u] |= (1 << v)
                final_bm[v] |= (1 << u)
            final_edges, final_score = _targeted_local_search(final_bm, n, max_iter=300, seed=1234, deadline=deadline)
            if final_score < best_score:
                best_score = final_score
                best_edges = final_edges
    
    return best_edges
# EVOLVE-BLOCK-END
