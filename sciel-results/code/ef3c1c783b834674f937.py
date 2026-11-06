import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] & adj[j]).bit_count() > 1:
                    return False
        return True

    def greedy_build(n, target_deg):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Uses a priority-based approach: always connect the lowest-degree vertex first.
        Returns adj list of bitmasks or None if failed.
        """
        adj = [0] * n
        deg = [0] * n
        
        # Priority queue simulation: repeatedly pick the lowest-degree vertex
        # and try to add as many edges as possible to it.
        
        for _ in range(n * target_deg):  # Max number of edge additions needed
            # Find vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])
            
            if deg[u] >= target_deg:
                # All vertices have at least target_deg
                if min(deg) >= target_deg:
                    return adj
                # But some might still be low, continue
                pass
            
            # Find all candidates v for u
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if not candidates:
                # Can't add any more edges to u, try next iteration
                # This vertex is "stuck" at current degree
                break
            
            # Add edge to the candidate with lowest degree (to balance)
            candidates.sort(key=lambda x: deg[x])
            v = candidates[0]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        if min(deg) >= target_deg:
            return adj
        return None

    def local_search(adj, n, target_deg, max_time=10.0):
        """
        Improve min degree using edge additions and swaps.
        Returns (adj, min_deg).
        """
        deg = [adj[i].bit_count() for i in range(n)]
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        no_improve = 0
        max_no_improve = 300
        
        while time.time() - start < max_time:
            if no_improve > max_no_improve:
                break
                
            # Find vertex with min degree
            u = min(range(n), key=lambda i: deg[i])
            current_min = deg[u]
            
            if current_min >= target_deg:
                if best_min_deg >= target_deg and check_valid(adj, n, target_deg):
                    return adj, best_min_deg
                # Try to polish further
                pass
            
            # Try to add an edge incident to u
            added = False
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                if (adj[u] & adj[v]) == 0:
                    candidates.append(v)
            
            if candidates:
                # Pick candidate that maximizes benefit: prefer low-degree v
                candidates.sort(key=lambda x: deg[x])
                v = candidates[0]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                added = True
            else:
                # No edge can be added to u directly.
                # Try a 2-edge swap: remove (u,w), add (u,v)
                # where removing (u,w) allows adding (u,v)
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    # Try each neighbor w to remove
                    success = False
                    for w in neighbors:
                        # Temporarily remove (u,w)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        deg[u] -= 1
                        deg[w] -= 1
                        
                        # Now check if we can add (u,v) for some v
                        for v in range(n):
                            if v == u:
                                continue
                            if adj[u] & (1 << v):
                                continue
                            if (adj[u] & adj[v]) == 0:
                                # Add (u,v)
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                deg[u] += 1
                                deg[v] += 1
                                success = True
                                break
                        
                        if success:
                            break
                        else:
                            # Revert removal of (u,w)
                            adj[u] |= (1 << w)
                            adj[w] |= (1 << u)
                            deg[u] += 1
                            deg[w] += 1
                
                if not success:
                    no_improve += 1
                    continue
            
            if added or success:
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1

        return best_adj, best_min_deg

    def polarity_seed(n):
        """
        Create a C4-free seed graph using a polarity-like construction.
        For n close to q^2+q+1, use the projective plane polarity graph structure.
        This gives a regular graph with good C4-free properties.
        """
        adj = [0] * n
        
        # Use a simple construction: connect i to i+1, i+2, ..., i+k mod n
        # where k is chosen to give desired degree while avoiding C4s.
        # For C4-freeness in circulant graphs, we need to be careful.
        
        # Alternative: use a greedy construction with a fixed seed
        # to get a deterministic high-quality start.
        
        # For now, just return None to use greedy_build
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        found = False
        
        # Strategy 1: Greedy Build with multiple seeds
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 123 + 456)
            adj = greedy_build(n, min_deg)
            
            if adj is not None and check_valid(adj, n, min_deg):
                edges = get_edges(adj, n)
                return {'n': n, 'edges': edges}
            
            # If greedy build didn't meet target, try local search
            if adj is not None:
                adj_ls, min_deg_ls = local_search(adj, n, min_deg, max_time=3.0)
                if min_deg_ls >= min_deg and check_valid(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 2: Random initialization + repair
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + 13)
            
            # Initialize with sparse random graph
            p = min_deg / n * 1.5
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase: remove C4s, then add edges to boost min degree
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                # Check for C4 violations
                violating_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] & adj[j]).bit_count() > 1:
                            violating_pairs.append((i, j))
                
                if violating_pairs:
                    # Fix one violation
                    i, j = random.choice(violating_pairs)
                    common = adj[i] & adj[j]
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove edges to reduce common neighbors to 1
                    for w in common_list[1:]:
                        # Remove edge from the higher-degree vertex
                        if adj[i].bit_count() >= adj[j].bit_count():
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                        else:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                else:
                    # No C4s, check min degree and add edges if needed
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            return {'n': n, 'edges': edges}
                        break
                    
                    # Add edges to low-degree vertices
                    low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
                    if not low_verts:
                        break
                    u = random.choice(low_verts)
                    candidates = []
                    for v in range(n):
                        if v == u:
                            continue
                        if not (adj[u] & (1 << v)):
                            if (adj[u] & adj[v]) == 0:
                                candidates.append(v)
                    
                    if candidates:
                        v = random.choice(candidates)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        break

    # Fallback: Return a valid small graph
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
