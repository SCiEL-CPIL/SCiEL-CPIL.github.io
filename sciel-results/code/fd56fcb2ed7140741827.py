import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    
    # Start with a known 4-chromatic triangle-free graph on 15 vertices (Mycielski of C7)
    def get_mycielski_c7():
        n_cycle = 7
        edges = []
        for i in range(n_cycle):
            edges.append([i, (i + 1) % n_cycle])
        for i in range(n_cycle):
            neighbors = [(i + 1) % n_cycle, (i - 1) % n_cycle]
            for nb in neighbors:
                edges.append([7 + i, nb])
        for i in range(n_cycle):
            edges.append([14, 7 + i])
        return edges
    
    base_edges = get_mycielski_c7()
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i > j:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True
    
    def is_not_3_colorable(adj, n):
        # Backtracking 3-coloring
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        # Try multiple orderings
        for order in [list(range(n)), sorted(range(n), key=lambda x: len(adj[x]), reverse=True)]:
            colors = [-1] * n
            def bt(idx):
                if idx == n:
                    return True
                v = order[idx]
                for c in range(3):
                    valid = True
                    for nb in adj[v]:
                        if colors[nb] == c:
                            valid = False
                            break
                    if valid:
                        colors[v] = c
                        if bt(idx + 1):
                            return True
                        colors[v] = -1
                return False
            if not bt(0):
                return True
        return False
    
    # Phase 1: Greedy edge addition
    # Try to add all possible edges that maintain triangle-freeness and 4-chromaticity
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Sort by some heuristic - maybe by sum of degrees to prioritize high-degree connections
    def edge_priority(e):
        u, v = e
        return len(adj[u]) + len(adj[v])
    
    # First, add edges that don't create triangles
    addable_edges = []
    for u, v in all_possible_edges:
        if v in adj[u]:
            continue
        # Check if adding (u,v) creates a triangle
        if not (adj[u] & adj[v]):
            addable_edges.append((u, v))
    
    # Sort by priority
    addable_edges.sort(key=edge_priority, reverse=True)
    
    for u, v in addable_edges:
        if time.time() > deadline:
            break
        if v in adj[u]:
            continue
        # Tentatively add edge
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if still 4-chromatic
        if is_not_3_colorable(adj, n):
            # Keep the edge
            pass
        else:
            # Remove the edge
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Phase 2: Local search - try swapping edges to increase count
    best_edges = sum(len(adj[i]) for i in range(n)) // 2
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing an edge and adding a different one
        current_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        random.shuffle(current_edges)
        
        for rem_u, rem_v in current_edges:
            if time.time() > deadline:
                break
            # Remove edge
            adj[rem_u].discard(rem_v)
            adj[rem_v].discard(rem_u)
            
            # Try to add a different edge
            candidates = [(i, j) for i in range(n) for j in range(i + 1, n)
                         if j not in adj[i] and not (adj[i] & adj[j])]
            random.shuffle(candidates)
            
            for add_u, add_v in candidates:
                if time.time() > deadline:
                    break
                # Add edge
                adj[add_u].add(add_v)
                adj[add_v].add(add_u)
                
                if is_not_3_colorable(adj, n):
                    # Check if we improved
                    new_edges = sum(len(adj[i]) for i in range(n)) // 2
                    if new_edges > best_edges:
                        best_edges = new_edges
                        improved = True
                        break
                    else:
                        # Keep it if same count (for diversity)
                        if new_edges == best_edges and random.random() < 0.3:
                            improved = True
                            break
                        else:
                            # Remove
                            adj[add_u].discard(add_v)
                            adj[add_v].discard(add_u)
            
            # If we didn't find a good swap, restore the removed edge
            if not any(adj[rem_u] & adj[rem_v]):  # Check if we're back to original state
                if (rem_u, rem_v) not in [(i, j) for i in range(n) for j in adj[i] if i < j]:
                    adj[rem_u].add(rem_v)
                    adj[rem_v].add(rem_u)
    
    # Build final edge list
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append([u, v])
    
    # Final validation
    if not is_triangle_free(adj, n) or not is_not_3_colorable(adj, n):
        # Fallback to base
        return base_edges
    
    return final_edges
