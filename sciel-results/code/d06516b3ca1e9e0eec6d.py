import time
import random
import itertools

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            common = adj_mask[a] & adj_mask[b]
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _is_k4_free_after_add(adj_mask, n, u, v):
    """Fast check: after adding edge (u,v), is graph still K4-free?"""
    common = adj_mask[u] & adj_mask[v]
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        if adj_mask[c] & common:
            return False
        c_bits ^= lb
    return True

def _is_triangle_free_mask(adj_mask, n, subset_mask):
    """Check if the set of vertices in subset_mask induces a triangle-free graph."""
    v_bits = subset_mask
    while v_bits:
        lb = v_bits & (-v_bits)
        u = lb.bit_length() - 1
        v_bits ^= lb
        
        N_u = adj_mask[u] & subset_mask
        w_bits = N_u
        while w_bits:
            lb2 = w_bits & (-w_bits)
            w = lb2.bit_length() - 1
            w_bits ^= lb2
            if adj_mask[w] & N_u:
                return False
    return True

def _max_tf_size_exact(adj_mask, n, timeout=0.5):
    """
    Find the largest triangle-free induced subgraph size using branch and bound.
    Optimized for n=18.
    """
    start_time = time.time()
    
    # Try sizes from n down to 1
    for k in range(n, 0, -1):
        if time.time() - start_time > timeout:
            break
            
        # Check all subsets of size k
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_triangle_free_mask(adj_mask, n, mask):
                return k
                
    return 1

def _find_large_tf_set(adj_mask, n, target_size=15):
    """
    Find a triangle-free set of size target_size.
    Returns the set as a bitmask or None.
    """
    for subset in itertools.combinations(range(n), target_size):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_triangle_free_mask(adj_mask, n, mask):
            return mask
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    n = 18
    all_edges = list(itertools.combinations(range(n), 2))
    
    def time_ok():
        return time.time() < deadline

    best_score = 18
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _max_tf_size_exact(adj_mask, n, timeout=1.0)
        return score

    # 1. Initial Construction: Random K4-free graph
    random.seed(42)
    
    # Generate a few random base graphs
    base_graphs = []
    for i in range(20):
        if not time_ok(): break
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_after_add(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        base_graphs.append(adj)
        
    # Evaluate base graphs
    for adj in base_graphs:
        if not time_ok(): break
        score = evaluate(adj)
        if score is not None and score < best_score:
            best_score = score
            best_adj_mask = adj[:]

    # 2. Directed Local Search
    current_adj = best_adj_mask[:]
    current_score = best_score
    
    for iteration in range(500):
        if not time_ok(): break
        
        tf_set_mask = _find_large_tf_set(current_adj, n, current_score)
        
        if tf_set_mask is None:
            new_score = evaluate(current_adj)
            if new_score is not None and new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_adj_mask = current_adj[:]
            continue
        
        verts = []
        m = tf_set_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb
        
        candidates = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u, v = verts[i], verts[j]
                if (current_adj[u] >> v) & 1:
                    continue
                
                common_in_s = current_adj[u] & current_adj[v] & tf_set_mask
                if common_in_s:
                    candidates.append((u, v))
        
        if not candidates:
            break
        
        u, v = random.choice(candidates)
        
        if _is_k4_free_after_add(current_adj, n, u, v):
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            new_score = evaluate(current_adj)
            if new_score is not None:
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_adj_mask = current_adj[:]
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        else:
            pass
            
    # 3. Final Polish: Try all single edge additions to best graph
    if time_ok():
        adj = best_adj_mask[:]
        improved = True
        while improved and time_ok():
            improved = False
            for u, v in all_edges:
                if not time_ok(): break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    score = evaluate(adj)
                    if score is not None and score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
