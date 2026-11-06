import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    core_n = 11
    
    # Grötzsch graph edges (Mycielski of C5)
    def _build_grotzsch_edges():
        edges = []
        # C5: 0-1-2-3-4-0
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([2, 3])
        edges.append([3, 4])
        edges.append([4, 0])
        # Mycielski: copies 5-9, top 10
        c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in c5_edges:
            edges.append([u, 5+v])
            edges.append([v, 5+u])
        for i in range(5):
            edges.append([10, 5+i])
        return edges

    core_edges = _build_grotzsch_edges()
    
    # Build core adjacency bitmask
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)
    
    # Find all independent sets in core, sorted by size descending
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind and mask != 0:
            vertices = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                vertices.append(u)
                m ^= lsb
            independent_sets.append(vertices)
    
    independent_sets.sort(key=lambda x: -len(x))
    
    # Take top K independent sets for search
    K = 30
    top_sets = independent_sets[:K]
    # Also include the empty set as an option
    top_sets.append([])  # empty set
    K_total = len(top_sets)
    
    # Precompute disjointness matrix for top sets
    D = [[False] * K_total for _ in range(K_total)]
    for i in range(K_total):
        si = set(top_sets[i])
        for j in range(K_total):
            sj = set(top_sets[j])
            D[i][j] = si.isdisjoint(sj)
    
    # Precompute max triangle-free edge count for each mask of the 10 possible new-new edges
    # Edges between 5 new vertices: pairs (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    # Index: 0:(0,1), 1:(0,2), 2:(0,3), 3:(0,4), 4:(1,2), 5:(1,3), 6:(1,4), 7:(2,3), 8:(2,4), 9:(3,4)
    pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    max_tf_lookup = [0] * (1 << 10)
    for mask in range(1 << 10):
        adj_new = [0] * 5
        for k in range(10):
            if mask & (1 << k):
                u, v = pairs[k]
                adj_new[u] |= (1 << v)
                adj_new[v] |= (1 << u)
        tf = True
        for u in range(5):
            for v in range(u + 1, 5):
                if adj_new[u] & (1 << v):
                    if adj_new[u] & adj_new[v]:
                        tf = False
                        break
            if not tf:
                break
        if tf:
            max_tf_lookup[mask] = bin(mask).count("1")
        else:
            # Find best subset of this mask that is triangle-free
            best = 0
            for sub in range(mask, -1, -1):
                if sub & ~mask != 0:
                    continue
                adj_sub = [0] * 5
                for k in range(10):
                    if sub & (1 << k):
                        u, v = pairs[k]
                        adj_sub[u] |= (1 << v)
                        adj_sub[v] |= (1 << u)
                tf2 = True
                for u2 in range(5):
                    for v2 in range(u2 + 1, 5):
                        if adj_sub[u2] & (1 << v2):
                            if adj_sub[u2] & adj_sub[v2]:
                                tf2 = False
                                break
                    if not tf2:
                        break
                if tf2:
                    cnt = bin(sub).count("1")
                    if cnt > best:
                        best = cnt
            max_tf_lookup[mask] = best
    
    # 3-colorability check with bitmasks
    def _is_3_colorable(adj, n):
        degrees = [bin(adj[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            for u in range(n):
                if color[u] != -1 and (adj[v] & (1 << u)):
                    used |= (1 << color[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def _evaluate(indices):
        """Evaluate a configuration of 5 independent sets for the new vertices.
        Returns (edge_count, is_4chromatic)."""
        # Build full adjacency
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        edge_count = len(core_edges)
        new_vertices = list(range(core_n, n))
        
        for pos in range(5):
            v_new = new_vertices[pos]
            s = top_sets[indices[pos]]
            for v_core in s:
                adj[v_new] |= (1 << v_core)
                adj[v_core] |= (1 << v_new)
                edge_count += 1
        
        # Determine possible new-new edges
        possible_mask = 0
        for k in range(10):
            i, j = pairs[k]
            if D[indices[i]][indices[j]]:
                possible_mask |= (1 << k)
        
        # Best TF subset
        best_count = max_tf_lookup[possible_mask]
        edge_count += best_count
        
        # Check 3-colorability
        if _is_3_colorable(adj, n):
            return (edge_count, False)
        else:
            return (edge_count, True)
    
    # Local search
    rng = random.Random(42)
    best_score = -1
    best_indices = None
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        indices = [rng.randint(0, K_total - 1) for _ in range(5)]
        score, valid = _evaluate(indices)
        if valid:
            current_score = score
        else:
            current_score = -1
        
        if current_score > best_score:
            best_score = current_score
            best_indices = list(indices)
        
        if current_score < 0:
            continue
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(5):
                for new_idx in range(K_total):
                    if new_idx == indices[i]:
                        continue
                    old_idx = indices[i]
                    indices[i] = new_idx
                    new_score, new_valid = _evaluate(indices)
                    if new_valid and new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_indices = list(indices)
                    else:
                        indices[i] = old_idx
    
    if best_indices is None:
        return core_edges
    
    # Reconstruct edges
    final_edges = list(core_edges)
    new_vertices = list(range(core_n, n))
    
    for pos in range(5):
        v_new = new_vertices[pos]
        s = top_sets[best_indices[pos]]
        for v_core in s:
            final_edges.append([v_new, v_core])
    
    # Add new-new edges
    i0, i1, i2, i3, i4 = best_indices
    mask = 0
    if D[i0][i1]: mask |= 1 << 0
    if D[i0][i2]: mask |= 1 << 1
    if D[i0][i3]: mask |= 1 << 2
    if D[i0][i4]: mask |= 1 << 3
    if D[i1][i2]: mask |= 1 << 4
    if D[i1][i3]: mask |= 1 << 5
    if D[i1][i4]: mask |= 1 << 6
    if D[i2][i3]: mask |= 1 << 7
    if D[i2][i4]: mask |= 1 << 8
    if D[i3][i4]: mask |= 1 << 9
    
    best_sub_mask = 0
    best_count = 0
    for sub_mask in range(1024):
        if sub_mask & ~mask != 0:
            continue
        if max_tf_lookup[sub_mask] > best_count:
            best_count = max_tf_lookup[sub_mask]
            best_sub_mask = sub_mask
    
    for k in range(10):
        if best_sub_mask & (1 << k):
            u, v = pairs[k]
            final_edges.append([new_vertices[u], new_vertices[v]])
    
    return final_edges
