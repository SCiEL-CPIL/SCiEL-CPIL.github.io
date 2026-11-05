import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    rng = random.Random(42)
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        Assumes (u,v) is not currently an edge.
        """
        for y in range(n):
            if y == u or y == v:
                continue
            ny = adj[y]
            if (ny >> v) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    if (nz >> v) & 1:
                        if bit_count(adj[u] & ny & nz) > 1:
                            return False
            
            if (ny >> u) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    if (nz >> u) & 1:
                        if bit_count(adj[v] & ny & nz) > 1:
                            return False
        
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_existing_edges(adj):
        """Return list of existing edges."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def get_missing_edges(adj):
        """Return list of missing edges."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    edges.append((i, j))
        return edges

    def systematic_swap_search(adj, cur_edges, all_edges, deadline):
        """
        Systematic swap search: try all pairs (existing, missing) in a deterministic order.
        Returns (new_adj, new_edge_count, found_improvement).
        """
        existing = get_existing_edges(adj)
        missing = get_missing_edges(adj)
        
        if not existing or not missing:
            return adj, cur_edges, False
        
        # Shuffle for variety but keep deterministic within a run
        rng.shuffle(existing)
        rng.shuffle(missing)
        
        improved = False
        for e1 in existing:
            if time.time() > deadline - 5:
                break
            e1_u, e1_v = e1
            for e2 in missing:
                if time.time() > deadline - 5:
                    break
                e2_u, e2_v = e2
                
                # Skip if same edge
                if (e1_u, e1_v) == (e2_u, e2_v) or (e1_u, e1_v) == (e2_v, e2_u):
                    continue
                
                # Try swap
                remove_edge(adj, e1_u, e1_v)
                if is_valid_add(adj, e2_u, e2_v):
                    add_edge(adj, e2_u, e2_v)
                    improved = True
                    # After one successful swap, re-evaluate
                    break
                else:
                    add_edge(adj, e1_u, e1_v)
            
            if improved:
                # Reset and continue
                break
        
        return adj, cur_edges, improved

    def repair_graph(adj):
        """
        Repair an invalid graph by removing edges until valid.
        Returns (adj, num_removed).
        """
        removed = 0
        changed = True
        while changed:
            changed = False
            # Check all triples
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        common = adj[a] & adj[b] & adj[c]
                        if bit_count(common) >= 3:
                            # Find an edge to remove from the common neighbors
                            # Remove edge involving one of the common neighbors
                            # Try removing edges incident to a, b, or c
                            for v in range(n):
                                if v in (a, b, c):
                                    continue
                                if (common >> v) & 1:
                                    # v is a common neighbor
                                    # Try removing (a,v), (b,v), or (c,v)
                                    for x in (a, b, c):
                                        if (adj[x] >> v) & 1:
                                            remove_edge(adj, x, v)
                                            removed += 1
                                            changed = True
                                            break
                                        if changed:
                                            break
                                    if changed:
                                        break
                            if changed:
                                break
                if changed:
                    break
            if changed:
                break
        
        return adj, removed

    best_edges = 0
    best_adj = None
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_starts = 60
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # 1. Greedy initialization with random order
        adj = [0] * n
        edge_order = all_edges[:]
        rng.shuffle(edge_order)
        for (u, v) in edge_order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        
        cur_edges = count_edges(adj)
        
        # 2. Local Search with multiple phases
        improved = True
        max_iterations = 20
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Phase 1: Add-only (try to add any missing edge)
            missing = get_missing_edges(adj)
            rng.shuffle(missing)
            added = False
            for (u, v) in missing:
                if time.time() > deadline - 5:
                    break
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    added = True
            
            if added:
                improved = True
            
            # Phase 2: Systematic swap search
            if time.time() > deadline - 5:
                break
            
            adj, cur_edges, swap_improved = systematic_swap_search(adj, cur_edges, all_edges, deadline)
            if swap_improved:
                improved = True
            
            # Phase 3: Repair + re-add (escape local optima)
            if not improved and time.time() > deadline - 10:
                break
            
            if not improved:
                # Try a more aggressive move: remove a random edge, then repair and re-add
                existing = get_existing_edges(adj)
                if existing:
                    e1_u, e1_v = rng.choice(existing)
                    remove_edge(adj, e1_u, e1_v)
                    cur_edges -= 1
                    
                    # Re-add greedily
                    missing = get_missing_edges(adj)
                    rng.shuffle(missing)
                    for (u, v) in missing:
                        if time.time() > deadline - 5:
                            break
                        if is_valid_add(adj, u, v):
                            add_edge(adj, u, v)
                            cur_edges += 1
                    
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = [a for a in adj]
                    
                    # If we got back to same or better, continue
                    if cur_edges >= best_edges:
                        improved = True
                    else:
                        # Try another random removal
                        existing = get_existing_edges(adj)
                        if existing:
                            e2_u, e2_v = rng.choice(existing)
                            remove_edge(adj, e2_u, e2_v)
                            cur_edges -= 1
                            
                            missing = get_missing_edges(adj)
                            rng.shuffle(missing)
                            for (u, v) in missing:
                                if time.time() > deadline - 5:
                                    break
                                if is_valid_add(adj, u, v):
                                    add_edge(adj, u, v)
                                    cur_edges += 1
                            
                            if cur_edges > best_edges:
                                best_edges = cur_edges
                                best_adj = [a for a in adj]
                            if cur_edges >= best_edges:
                                improved = True
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
        
        if not improved:
            # Try one more aggressive restart with different random seed
            adj2 = [0] * n
            edge_order2 = all_edges[:]
            rng.shuffle(edge_order2)
            for (u, v) in edge_order2:
                if is_valid_add(adj2, u, v):
                    add_edge(adj2, u, v)
            
            cur_edges2 = count_edges(adj2)
            
            # Quick local search
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                
                missing = get_missing_edges(adj2)
                rng.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_add(adj2, u, v):
                        add_edge(adj2, u, v)
                        cur_edges2 += 1
                
                adj2, cur_edges2, _ = systematic_swap_search(adj2, cur_edges2, all_edges, deadline)
            
            if cur_edges2 > best_edges:
                best_edges = cur_edges2
                best_adj = [a for a in adj2]
                
    if best_adj is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
