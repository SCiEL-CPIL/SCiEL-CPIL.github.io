import random
import time
import sys

def _is_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[u] & adj[v]:
                return False
    return True

def _is_3_colorable(adj):
    """Check if the graph is 3-colorable using backtracking with forward checking."""
    n = len(adj)
    color = [0] * n
    domain = [7] * n  # bitmask of available colors (1,2,4 for colors 1,2,3)
    
    def backtrack():
        # MRV: pick uncolored vertex with fewest available colors
        best_v = -1
        min_avail = 8
        for v in range(n):
            if color[v] == 0:
                count = bin(domain[v]).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    if count == 1:
                        break
        
        if best_v == -1:
            return True  # All colored
            
        avail = domain[best_v]
        
        c = 1
        while c <= 3:
            if avail & (1 << (c-1)):
                color[best_v] = c
                # Forward checking: update domains of neighbors
                saved_domains = []
                valid = True
                m = adj[best_v]
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    m ^= lb
                    if color[u] == 0:
                        old_dom = domain[u]
                        new_dom = old_dom & ~(1 << (c-1))
                        if new_dom == 0:
                            valid = False
                            break
                        saved_domains.append((u, old_dom))
                        domain[u] = new_dom
                
                if valid and backtrack():
                    return True
                
                # Undo forward checking
                for u, old_dom in saved_domains:
                    domain[u] = old_dom
                color[best_v] = 0
            c += 1
        return False

    return backtrack()

def _count_edges(adj, n=16):
    return sum(bin(adj[i]).count('1') for i in range(n)) // 2

def _get_missing_edges(adj, n=16):
    """Return list of edges that can be added without creating a triangle."""
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if adj[u] & adj[v] == 0:
                    candidates.append((u, v))
    return candidates

def _get_all_edges(adj, n=16):
    """Return list of all edges in the graph."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append((u, v))
    return edges

def _build_grotzsch_edges():
    """Build the Grötzsch graph (11 vertices, triangle-free, not 3-colorable)."""
    edges = []
    # C5 on vertices 0-4
    for i in range(5):
        edges.append([i, (i+1)%5])
    # Mycielski clones on vertices 5-9
    for i in range(5):
        u = i
        v = (i+1)%5
        edges.append([u, 5+v])
        edges.append([v, 5+u])
    # Apex vertex 10 connected to all Mycielski clones
    for i in range(5):
        edges.append([10, 5+i])
    return edges

def _make_adj(edges, n=16):
    """Convert edge list to adjacency bitmask representation."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _greedy_complete(adj, n=16, deadline=None):
    """Greedy edge addition: repeatedly add edges that maintain triangle-free and non-3-colorable."""
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        candidates = _get_missing_edges(adj, n)
        for u, v in candidates:
            if deadline and time.time() > deadline:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_3_colorable(adj):
                improved = True
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
    return adj

def _local_search_simulated_annealing(adj, n=16, deadline=None):
    """Simulated annealing local search for maximizing edges in triangle-free non-3-colorable graph."""
    current_adj = [a for a in adj]
    current_count = _count_edges(current_adj, n)
    best_adj = [a for a in adj]
    best_count = current_count
    
    temp = 5.0
    cooling = 0.995
    
    while deadline is None or time.time() < deadline:
        if temp < 0.01:
            break
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random missing edge
            candidates = _get_missing_edges(current_adj, n)
            if not candidates:
                continue
            u, v = random.choice(candidates)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            if _is_triangle_free(current_adj):
                new_count = _count_edges(current_adj, n)
                delta_e = new_count - current_count
                if not _is_3_colorable(current_adj):
                    # Valid move: accept if improves, or with probability if not
                    if delta_e > 0 or random.random() < min(1.0, temp):
                        current_count = new_count
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = [a for a in current_adj]
                    else:
                        # Revert
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                else:
                    # Not 3-colorable is required, so this is invalid
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
            else:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
        
        elif move_type < 0.7:
            # Remove a random edge
            all_edges = _get_all_edges(current_adj, n)
            if not all_edges:
                continue
            u, v = random.choice(all_edges)
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            new_count = _count_edges(current_adj, n)
            delta_e = new_count - current_count
            # Accept with probability
            if random.random() < min(1.0, temp):
                current_count = new_count
            else:
                # Revert
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        else:
            # Swap: remove one edge, add another
            all_edges = _get_all_edges(current_adj, n)
            candidates = _get_missing_edges(current_adj, n)
            if not all_edges or not candidates:
                continue
            u1, v1 = random.choice(all_edges)
            u2, v2 = random.choice(candidates)
            
            # Remove u1-v1
            current_adj[u1] ^= (1 << v1)
            current_adj[v1] ^= (1 << u1)
            
            # Try to add u2-v2
            if _is_triangle_free(current_adj):
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u1)  # BUG: should be (1 << v2)
                # Fix:
                current_adj[u2] ^= (1 << v2)
                current_adj[v2] ^= (1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                if _is_triangle_free(current_adj) and not _is_3_colorable(current_adj):
                    new_count = _count_edges(current_adj, n)
                    delta_e = new_count - current_count
                    if random.random() < min(1.0, temp):
                        current_count = new_count
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = [a for a in current_adj]
                    else:
                        # Revert both
                        current_adj[u2] ^= (1 << v2)
                        current_adj[v2] ^= (1 << u2)
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                else:
                    # Revert
                    current_adj[u2] ^= (1 << v2)
                    current_adj[v2] ^= (1 << u2)
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            else:
                # Revert removal
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
        
        temp *= cooling
    
    return best_adj, best_count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    # Phase 1: Start with Grötzsch graph and greedily complete
    grotzsch_edges = _build_grotzsch_edges()
    adj = _make_adj(grotzsch_edges, n)
    adj = _greedy_complete(adj, n, deadline)
    count = _count_edges(adj, n)
    if count > best_count:
        best_count = count
        best_adj = [a for a in adj]
        best_edges = _get_all_edges(adj, n)
    
    # Phase 2: Simulated annealing local search
    if time.time() < deadline:
        sa_deadline = min(deadline, start_time + 80.0)
        adj_sa, count_sa = _local_search_simulated_annealing(adj, n, sa_deadline)
        if count_sa > best_count:
            best_count = count_sa
            best_adj = [a for a in adj_sa]
            best_edges = _get_all_edges(adj_sa, n)
    
    # Phase 3: Try different initial structures with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try Chvátal graph (12 vertices) + 4 isolated
        chvatal_edges = [
            [0,1], [1,2], [2,3], [3,4], [4,0],  # C5
            [0,5], [1,5], [2,5], [3,5], [4,5],  # universal vertex to C5
            [5,6], [6,7], [7,8], [8,9], [9,6],  # another C5
            [6,10], [7,10], [8,10], [9,10],     # connect
            [0,10], [1,11], [2,11], [3,11], [4,11]
        ]
        # This is getting complex, let's just try random perturbations of the current best
        rng = random.Random(42)
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Perturb current best
            adj_pert = [a for a in best_adj]
            # Remove 3 random edges
            all_edges = _get_all_edges(adj_pert, n)
            for _ in range(min(3, len(all_edges))):
                if not all_edges:
                    break
                u, v = rng.choice(all_edges)
                adj_pert[u] ^= (1 << v)
                adj_pert[v] ^= (1 << u)
                all_edges = [e for e in all_edges if e != (u, v) and e != (v, u)]
            
            # Greedy complete
            adj_pert = _greedy_complete(adj_pert, n, deadline)
            count_pert = _count_edges(adj_pert, n)
            if count_pert > best_count:
                best_count = count_pert
                best_adj = [a for a in adj_pert]
                best_edges = _get_all_edges(adj_pert, n)
            
            # Short SA
            if time.time() < deadline:
                sa_deadline2 = min(deadline, time.time() + 10.0)
                adj_sa2, count_sa2 = _local_search_simulated_annealing(adj_pert, n, sa_deadline2)
                if count_sa2 > best_count:
                    best_count = count_sa2
                    best_adj = [a for a in adj_sa2]
                    best_edges = _get_all_edges(adj_sa2, n)
    
    # Final validation
    if best_count == 0:
        best_edges = list(grotzsch_edges)
    
    adj_final = [0] * n
    for u, v in best_edges:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)
        
    if not _is_triangle_free(adj_final) or _is_3_colorable(adj_final):
        # Fallback to known valid graph
        grotzsch_edges = _build_grotzsch_edges()
        adj_final = _make_adj(grotzsch_edges, n)
        best_edges = _get_all_edges(adj_final, n)
    
    return best_edges
