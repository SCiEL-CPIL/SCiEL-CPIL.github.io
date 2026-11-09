"""Construct a C4-free graph with high minimum degree using structured seeding + local search."""
import time
import random

def _is_c4_free(n, adj_bitsets):
    """Check if graph is C4-free using bitsets."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bitsets[u] & adj_bitsets[v]
            # Count bits
            c = common
            cnt = 0
            while c:
                cnt += 1
                c &= c - 1
            if cnt >= 2:
                return False
    return True

def _min_degree_from_bitsets(n, adj_bitsets):
    """Compute minimum degree from bitsets."""
    min_deg = n
    for i in range(n):
        deg = bin(adj_bitsets[i]).count('1')
        if deg < min_deg:
            min_deg = deg
    return min_deg

def _creates_c4(adj, u, v):
    """Check if adding edge (u,v) would create a C4. Returns True if it does."""
    A = adj[v] & ~(1 << u)
    B = adj[u] & ~(1 << v)
    a_mask = A
    while a_mask:
        w_bit = a_mask & (-a_mask)
        w = w_bit.bit_length() - 1
        a_mask ^= w_bit
        if adj[w] & B & ~(1 << w):
            return True
    return False

def _try_add_edge(adj, deg, u, v):
    """Try to add edge (u,v). Returns True if added successfully."""
    if u == v or (adj[u] >> v) & 1:
        return False
    if _creates_c4(adj, u, v):
        return False
    adj[u] |= 1 << v
    adj[v] |= 1 << u
    deg[u] += 1
    deg[v] += 1
    return True

def _try_remove_edge(adj, deg, u, v):
    """Remove edge (u,v). Returns True if removed."""
    if not ((adj[u] >> v) & 1):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_n = 39
    best_edges = []
    
    def elapsed():
        return time.time() - start_time
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if elapsed() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found = False
        # Try multiple random seeds with structured initialization
        for seed in range(500):
            if elapsed() > deadline:
                break
            
            random.seed(seed)
            adj = [0] * target_n
            deg = [0] * target_n
            
            # Phase 1: Build initial graph with random edges, ensuring C4-free
            # Start with a sparse random graph, then add edges greedily
            # Priority: favor edges incident to lower-degree vertices
            
            # Generate all possible edges with priority based on current degree
            all_edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    all_edges.append((u, v))
            random.shuffle(all_edges)
            
            # Greedy edge addition with fast C4 check, prioritizing low-degree vertices
            for u, v in all_edges:
                if elapsed() > deadline:
                    break
                
                # Prioritize edges where at least one endpoint has low degree
                # (This is a heuristic - we process in shuffled order but could sort)
                if _try_add_edge(adj, deg, u, v):
                    pass  # Edge added
            
            # Phase 2: Local search to improve minimum degree
            # Try to increase minimum degree by adding edges or swapping
            
            # First check current min degree
            cur_min_deg = min(deg)
            
            # If min degree is already sufficient, verify and accept
            if cur_min_deg >= min_deg_needed:
                # Verify C4-free
                if _is_c4_free(target_n, adj):
                    edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            a_mask ^= v_bit
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            # If not sufficient, try local search: add edges and swap edges
            # Edge swaps: remove one edge, add another, maintaining C4-freeness
            for iteration in range(300):
                if elapsed() > deadline:
                    break
                
                # Find vertices with minimum degree
                min_d = min(deg)
                if min_d >= min_deg_needed:
                    break
                
                # Strategy 1: Try to add an edge incident to a low-degree vertex
                low_verts = [i for i in range(target_n) if deg[i] <= min_d + 1]
                random.shuffle(low_verts)
                
                improved = False
                for u in low_verts:
                    if elapsed() > deadline:
                        break
                    
                    # Try all possible edges from u
                    candidates = []
                    for v in range(target_n):
                        if v != u and not (adj[u] >> v) & 1:
                            candidates.append(v)
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        if _try_add_edge(adj, deg, u, v):
                            improved = True
                            break
                    if improved:
                        break
                
                # Strategy 2: If no simple add worked, try edge swaps
                if not improved:
                    # Pick a random existing edge to potentially remove
                    existing_edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                existing_edges.append((u, v))
                            a_mask ^= v_bit
                    
                    if existing_edges:
                        random.shuffle(existing_edges)
                        for (ru, rv) in existing_edges[:20]:  # Try up to 20 swaps
                            if elapsed() > deadline:
                                break
                            
                            # Try to find a replacement edge
                            # Remove (ru, rv) temporarily
                            _try_remove_edge(adj, deg, ru, rv)
                            
                            # Now try to add edges to improve min degree
                            min_d2 = min(deg)
                            low_verts2 = [i for i in range(target_n) if deg[i] <= min_d2 + 1]
                            random.shuffle(low_verts2)
                            
                            added_back = False
                            for u2 in low_verts2[:5]:
                                if elapsed() > deadline:
                                    break
                                
                                candidates2 = []
                                for v2 in range(target_n):
                                    if v2 != u2 and not (adj[u2] >> v2) & 1:
                                        candidates2.append(v2)
                                random.shuffle(candidates2)
                                
                                for v2 in candidates2:
                                    if _try_add_edge(adj, deg, u2, v2):
                                        added_back = True
                                        break
                                if added_back:
                                    break
                            
                            if not added_back:
                                # Re-add the removed edge
                                _try_add_edge(adj, deg, ru, rv)
                
                if not improved:
                    # Check if we made progress via swaps
                    if min(deg) >= min_d:
                        # No improvement, break
                        if min(deg) < min_deg_needed:
                            break
            
            # After local search, check again
            cur_min_deg = min(deg)
            if cur_min_deg >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            a_mask ^= v_bit
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            
            # Additional polish: try to boost min degree further with targeted additions
            if not found:
                for polish_iter in range(50):
                    if elapsed() > deadline:
                        break
                    
                    cur_min = min(deg)
                    if cur_min >= min_deg_needed:
                        break
                    
                    # Focus on the lowest-degree vertex
                    lowest = min(range(target_n), key=lambda i: deg[i])
                    
                    # Try to add edges from lowest to non-neighbors
                    candidates = [v for v in range(target_n) if v != lowest and not (adj[lowest] >> v) & 1]
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        if _try_add_edge(adj, deg, lowest, v):
                            break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Return a trivial valid graph
        return {'n': 39, 'edges': []}
