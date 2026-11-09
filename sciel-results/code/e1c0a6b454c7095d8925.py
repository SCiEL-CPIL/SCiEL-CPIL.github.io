import random
import time

def _popcount(x):
    return bin(x).count('1')

def _greedy_max_triangle_free(adj):
    """Fast greedy upper bound for max triangle-free induced subgraph."""
    n = 18
    full_mask = (1 << n) - 1
    best = 0
    # Try multiple orderings
    for perm_seed in range(3):
        rng = random.Random(perm_seed)
        order = list(range(n))
        rng.shuffle(order)
        current = 0
        size = 0
        for v in order:
            bit = 1 << v
            if current & adj[v]:
                # Check if adding v creates a triangle
                nb = current & adj[v]
                # Check if nb contains any edge
                has_edge = False
                m = nb
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    if adj[u] & nb:
                        has_edge = True
                        break
                if has_edge:
                    continue
            current |= bit
            size += 1
        if size > best:
            best = size
    return best

def _is_k4_free_bitmask(adj):
    """Check if graph with adjacency bitmasks is K4-free."""
    n = 18
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                m = common
                while m:
                    lsb = m & -m
                    a = lsb.bit_length() - 1
                    m ^= lsb
                    if adj[a] & common:
                        return False
    return True

def _max_triangle_free_bitmask(adj):
    """
    Find the maximum number of vertices inducing a triangle-free subgraph.
    Uses branch and bound with bitmasks, greedy upper bound, and good ordering.
    """
    n = 18
    full_mask = (1 << n) - 1
    best = [0]
    
    # Get a good initial lower bound via greedy
    greedy_lb = _greedy_max_triangle_free(adj)
    best[0] = greedy_lb
    
    # Precompute: for each pair (u,v) that are adjacent, the common neighbors
    # This helps with fast triangle detection
    # adj_common[u] = bitmask of vertices adjacent to both u and some neighbor... 
    # Actually let's precompute: for each v, adj[v] is the bitmask of neighbors
    
    def _has_edge_in_mask(mask):
        """Check if the subgraph induced by mask contains any edge. Fast path for small masks."""
        if not mask:
            return False
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if adj[v] & mask:
                return True
        return False
    
    def _greedy_upper_bound(candidates, current_size):
        """Quick greedy upper bound: try to add as many as possible greedily."""
        # This is a fast heuristic upper bound
        remaining = candidates
        count = 0
        # Sort by degree in remaining (approximate)
        while remaining and count <= 18 - current_size - best[0]:
            # Pick vertex with min degree in remaining (more likely to be addable)
            best_v = -1
            best_deg = 999
            m = remaining
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = _popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            remaining &= ~(1 << best_v)
            count += 1
            if current_size + count <= best[0]:
                return current_size + count
        return current_size + count
    
    def branch(candidates, current_set, current_size):
        # Pruning with simple bound
        if current_size + _popcount(candidates) <= best[0]:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidate subgraph (for better pruning)
        v_idx = -1
        best_score = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Score: how many neighbors in candidates (higher = more constraints)
            deg = _popcount(adj[v] & candidates)
            if deg > best_score:
                best_score = deg
                v_idx = v
        
        v = v_idx
        v_bit = 1 << v
        rest_candidates = candidates & ~v_bit
        
        # Option 1: Include v (try this first if it's feasible, for better pruning)
        N_v = adj[v] & current_set
        can_add = True
        if N_v and _has_edge_in_mask(N_v):
            can_add = False
        
        if can_add:
            new_set = current_set | v_bit
            new_size = current_size + 1
            
            # Update candidates: remove vertices that would form triangle with v and some u in current_set
            forbidden = 0
            if N_v:
                m3 = N_v
                while m3:
                    lsb3 = m3 & -m3
                    u = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    forbidden |= (adj[v] & adj[u])
            
            new_candidates = rest_candidates & ~forbidden
            branch(new_candidates, new_set, new_size)
        
        # Option 2: Exclude v
        branch(rest_candidates, current_set, current_size)
    
    branch(full_mask, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Initial graph: Turan T(18,3)
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_score = _max_triangle_free_bitmask(adj)
    best_adj = adj[:]
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges
        
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if not (m & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    # Systematic local search: try all single edge flips
    def systematic_local_search(adj_current, score_current):
        adj_work = adj_current[:]
        score_work = score_current
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            edges = get_edges(adj_work)
            non_edges = get_non_edges(adj_work)
            
            # Try removing edges
            for u, v in edges:
                if time.time() > deadline - 10:
                    break
                adj_work[u] &= ~(1 << v)
                adj_work[v] &= ~(1 << u)
                # Fast pre-check: greedy upper bound
                greedy_ub = _greedy_max_triangle_free(adj_work)
                if greedy_ub < score_work:
                    s = _max_triangle_free_bitmask(adj_work)
                    if s < score_work:
                        score_work = s
                        improved = True
                    else:
                        adj_work[u] |= (1 << v)
                        adj_work[v] |= (1 << u)
                else:
                    # Greedy says it could be at least score_work, but let's still check
                    # if it's clearly not better, skip
                    s = _max_triangle_free_bitmask(adj_work)
                    if s < score_work:
                        score_work = s
                        improved = True
                    else:
                        adj_work[u] |= (1 << v)
                        adj_work[v] |= (1 << u)
            
            # Try adding edges
            for u, v in non_edges:
                if time.time() > deadline - 10:
                    break
                adj_work[u] |= (1 << v)
                adj_work[v] |= (1 << u)
                if _is_k4_free_bitmask(adj_work):
                    greedy_ub = _greedy_max_triangle_free(adj_work)
                    if greedy_ub < score_work:
                        s = _max_triangle_free_bitmask(adj_work)
                        if s < score_work:
                            score_work = s
                            improved = True
                        else:
                            adj_work[u] &= ~(1 << v)
                            adj_work[v] &= ~(1 << u)
                    else:
                        s = _max_triangle_free_bitmask(adj_work)
                        if s < score_work:
                            score_work = s
                            improved = True
                        else:
                            adj_work[u] &= ~(1 << v)
                            adj_work[v] &= ~(1 << u)
                else:
                    adj_work[u] &= ~(1 << v)
                    adj_work[v] &= ~(1 << u)
        
        return adj_work, score_work
    
    # First: systematic local search from Turan
    best_adj, best_score = systematic_local_search(best_adj, best_score)
    
    # Then: random restarts with local search
    attempts = 0
    while time.time() < deadline - 5:
        # Strategy 1: Greedy construction - start with a random ordering and add edges greedily
        adj_new = [0] * n
        rng = random.Random(attempts * 7 + 13)
        order = list(range(n))
        rng.shuffle(order)
        
        # Build a K4-free graph by adding edges in a structured way
        # Start with a random partial coloring approach
        # Assign each vertex a "color" from 3 colors, then add edges between different colors
        # but randomly skip some to avoid K4
        colors = [0] * n
        for i in range(n):
            colors[order[i]] = i % 3
        
        # Add edges between different color classes with some probability
        p = 0.7 + rng.random() * 0.3  # probability of adding edge
        for i in range(n):
            for j in range(i + 1, n):
                if colors[i] != colors[j] and rng.random() < p:
                    adj_new[i] |= (1 << j)
                    adj_new[j] |= (1 << i)
        
        # Remove edges that create K4
        # Iteratively remove edges that are part of a K4
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_new[i] & (1 << j):
                        # Check if (i,j) is part of a K4
                        common = adj_new[i] & adj_new[j] & ~((1 << i) | (1 << j))
                        m = common
                        while m:
                            lsb = m & -m
                            a = lsb.bit_length() - 1
                            m ^= lsb
                            if adj_new[a] & common:
                                # Found K4, remove edge (i,j)
                                adj_new[i] &= ~(1 << j)
                                adj_new[j] &= ~(1 << i)
                                changed = True
                                break
                        if changed:
                            break
                if changed:
                    break
        
        # Do a quick local search from this constructed graph
        score_init = _max_triangle_free_bitmask(adj_new)
        adj_new, score_new = systematic_local_search(adj_new, score_init)
        
        if score_new < best_score:
            best_score = score_new
            best_adj = adj_new[:]
        
        # Also try random swaps from current best
        if time.time() < deadline - 5:
            adj_try = best_adj[:]
            edges = get_edges(adj_try)
            non_edges = get_non_edges(adj_try)
            if edges and non_edges:
                for _ in range(min(30, len(edges))):
                    if time.time() > deadline - 5:
                        break
                    u, v = random.choice(edges)
                    x, y = random.choice(non_edges)
                    adj_try[u] &= ~(1 << v)
                    adj_try[v] &= ~(1 << u)
                    adj_try[x] |= (1 << y)
                    adj_try[y] |= (1 << x)
                    if _is_k4_free_bitmask(adj_try):
                        greedy_ub = _greedy_max_triangle_free(adj_try)
                        if greedy_ub < best_score:
                            s = _max_triangle_free_bitmask(adj_try)
                            if s < best_score:
                                best_score = s
                                best_adj = adj_try[:]
                    # Revert
                    adj_try[u] |= (1 << v)
                    adj_try[v] |= (1 << u)
                    adj_try[x] &= ~(1 << y)
                    adj_try[y] &= ~(1 << x)
                    # Rebuild lists
                    edges = get_edges(adj_try)
                    non_edges = get_non_edges(adj_try)
        
        attempts += 1

    # Convert best_adj back to list of edges
    final_edges = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                final_edges.append([i, j])
            m ^= lsb
            
    return final_edges
