import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 8000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            # Helper: check if adding (u,v) creates a C4
            # Adding edge (u,v) creates C4 iff ∃w∈N(u) such that N(w)∩N(v) ≠ ∅
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False
                # Check all neighbors of u: for each w in N(u), check if adj[w] & adj[v] != 0
                m = adj[u]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if adj[w] & adj[v]:
                        return False
                    m ^= lsb
                return True
            
            # --- Polarity seed: structured initialization ---
            # Split vertices into two groups (polarities)
            # Connect vertices within/across groups to build high minimum degree
            # Strategy: random partition into two sets, connect all pairs across
            # This gives a bipartite-like structure which is C4-free if we're careful
            # Actually bipartite graphs CAN have C4. Let's use a different approach.
            # 
            # Better: Use a "blow-up" of a C4-free base graph.
            # Or: Start with a random graph biased toward high degree, then repair.
            #
            # Simplest effective seed: random permutation, connect each vertex
            # to a random subset of others, ensuring no C4 via can_add check.
            
            # Use different seed strategies based on restart number
            if restart % 3 == 0:
                # Strategy A: Greedy from empty with random order
                order = list(range(n))
                random.shuffle(order)
                for u in order:
                    # Try to connect u to min_deg other vertices
                    candidates = [v for v in order if v != u and not ((adj[u] >> v) & 1)]
                    random.shuffle(candidates)
                    for v in candidates:
                        if degrees[u] >= min_deg:
                            break
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
            elif restart % 3 == 1:
                # Strategy B: Bipartite-style seed
                # Split into two groups, connect across but check C4
                perm = list(range(n))
                random.shuffle(perm)
                group_a = perm[:n // 2]
                group_b = perm[n // 2:]
                for u in group_a:
                    for v in group_b:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
            else:
                # Strategy C: Random dense seed with C4 checking
                all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                random.shuffle(all_pairs)
                for u, v in all_pairs:
                    if can_add(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                    if time.time() > deadline:
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                # Verify C4-free
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            ai = adj[i] & ~(1 << j)
                            aj = adj[j] & ~(1 << i)
                            m = ai
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                if adj[a] & aj:
                                    c4_free = False
                                    break
                                m ^= lsb
                if c4_free:
                    return {'n': n, 'edges': edges}
            
            # Phase 1: Greedy edge addition to fix deficient vertices
            if time.time() < deadline:
                added_any = True
                while added_any and time.time() < deadline:
                    added_any = False
                    
                    # Find vertices below target degree
                    low_verts = [i for i in range(n) if degrees[i] < min_deg]
                    
                    if not low_verts:
                        break
                    
                    # Most constrained first (fewest non-neighbors)
                    u = min(low_verts, key=lambda x: n - 1 - degrees[x])
                    
                    # Try non-neighbors, prefer lower-degree vertices
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    if not non_neighbors:
                        break
                    non_neighbors.sort(key=lambda v: degrees[v])
                    
                    for v in non_neighbors:
                        if can_add(u, v):
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            added_any = True
                            break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            ai = adj[i] & ~(1 << j)
                            aj = adj[j] & ~(1 << i)
                            m = ai
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                if adj[a] & aj:
                                    c4_free = False
                                    break
                                m ^= lsb
                if c4_free:
                    return {'n': n, 'edges': edges}
            
            # Phase 2: Local search with edge swaps
            if time.time() < deadline:
                for ls_iter in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Find deficient vertices
                    deficient = [i for i in range(n) if degrees[i] < min_deg]
                    if not deficient:
                        break
                    
                    # Most constrained deficient vertex
                    u = min(deficient, key=lambda x: n - 1 - degrees[x])
                    
                    # Find potential partners for u
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    non_neighbors.sort(key=lambda v: degrees[v])
                    
                    improved = False
                    for v in non_neighbors:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            improved = True
                            break
                    
                    if not improved:
                        # Try edge swap: remove an edge (a,b) where a or b is u's non-neighbor
                        # and add (u, v) where v was a's or b's neighbor
                        # This is complex; try a simpler approach:
                        # Randomly remove an edge incident to a high-degree vertex
                        # and try to add edges to deficient vertices
                        if random.random() < 0.3:
                            # Pick a random edge to potentially remove
                            high_deg_verts = [i for i in range(n) if degrees[i] > min_deg]
                            if high_deg_verts:
                                a = random.choice(high_deg_verts)
                                neighbors_a = [x for x in range(n) if x != a and ((adj[a] >> x) & 1)]
                                if neighbors_a:
                                    b = random.choice(neighbors_a)
                                    # Try removing (a,b) and adding (u, v) for some v
                                    # Check if removing (a,b) allows adding (u, something)
                                    # For now, just remove and hope greedy can fix
                                    adj[a] &= ~(1 << b)
                                    adj[b] &= ~(1 << a)
                                    degrees[a] -= 1
                                    degrees[b] -= 1
                                    edges.remove([a, b]) if [a, b] in edges else (edges.remove([b, a]) if [b, a] in edges else None)
                                    improved = True
            
            # Final check
            if min(degrees) >= min_deg:
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            ai = adj[i] & ~(1 << j)
                            aj = adj[j] & ~(1 << i)
                            m = ai
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                if adj[a] & aj:
                                    c4_free = False
                                    break
                                m ^= lsb
                if c4_free:
                    return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    # For n=39, min_deg=0, empty graph works
    return {'n': 39, 'edges': []}
