import time

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # Rim edges (Hamiltonian cycle)
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
    # Chord edges: (0, 2k) for k=1..19
    chord_edges = [(0, 2 * k) for k in range(1, N // 2)]
    
    edges = rim_edges + chord_edges
    
    # De-duplicate and normalize
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    
    edge_set = set()
    adj = {i: set() for i in range(N)}
    for a, b in edges:
        u, v = min(a, b), max(a, b)
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    # Generate initial cycles
    cycles = {}
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L % 2 == 1:
            m = (L - 1) // 2
            cyc = list(range(0, 2 * m + 1))
        else:
            m = L // 2
            if m == 1:
                continue
            cyc = [0] + list(range(2, 2 * m + 1))
        cycles[L] = cyc
    
    # Validate initial cycles
    ok = True
    for L in range(3, N + 1):
        cyc = cycles[L]
        if len(set(cyc)) != len(cyc):
            ok = False
            break
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                ok = False
                break
        if not ok:
            break
    
    if not ok:
        # Fallback
        return {'edges': edges, 'cycles': [cycles[L] for L in range(3, N + 1)]}
    
    # Greedy edge removal: try to remove edges not critical for any cycle
    # Sort edges: prioritize removing chords first (they're more "optional")
    candidate_edges = list(edge_set)
    def edge_sort_key(e):
        u, v = e
        if u == 0 or v == 0:
            return (0, abs(u - v))
        else:
            return (1, 0)
    candidate_edges.sort(key=edge_sort_key)
    
    def find_cycle_of_length(L, adj, edge_set):
        """Find a simple cycle of length L using DFS with pruning."""
        if L < 3 or L > N:
            return None
        
        for start in range(N):
            if len(adj[start]) < 2:
                continue
            visited = set()
            path = []
            
            def dfs(u, depth):
                if depth == L:
                    if start in adj[u]:
                        return path[:]
                    return None
                if depth > L:
                    return None
                
                visited.add(u)
                path.append(u)
                
                for v in sorted(adj[u]):
                    if v == start and depth == L - 1:
                        return path[:]
                    if v not in visited:
                        res = dfs(v, depth + 1)
                        if res:
                            return res
                
                path.pop()
                visited.remove(u)
                return None
            
            res = dfs(start, 0)
            if res:
                return res
        return None
    
    # Try removing edges one by one
    for e in candidate_edges:
        if time.time() - t0 > DEADLINE_SECONDS - 5:
            break
            
        u, v = e
        if e not in edge_set:
            continue
        
        # Check if this edge is used in any current cycle
        used_in_cycles = []
        for L in range(3, N + 1):
            cyc = cycles[L]
            for i in range(len(cyc)):
                c_u = cyc[i]
                c_v = cyc[(i + 1) % len(cyc)]
                if (min(c_u, c_v), max(c_u, c_v)) == e:
                    used_in_cycles.append(L)
                    break
        
        if not used_in_cycles:
            # Edge not in any certificate - safe to remove
            edge_set.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            continue
        
        # Edge is in some cycles - try to find alternative cycles
        # Temporarily remove edge
        edge_set.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Check if all cycles can still be formed
        all_valid = True
        new_cycles = {}
        for L in range(3, N + 1):
            if L not in used_in_cycles:
                new_cycles[L] = cycles[L]
            else:
                # Need to find new cycle of length L
                new_cyc = find_cycle_of_length(L, adj, edge_set)
                if new_cyc is None:
                    all_valid = False
                    break
                new_cycles[L] = new_cyc
        
        if all_valid:
            # Keep the edge removed, update cycles
            cycles = new_cycles
        else:
            # Restore edge
            edge_set.add(e)
            adj[u].add(v)
            adj[v].add(u)
    
    # Final validation
    final_ok = True
    for L in range(3, N + 1):
        cyc = cycles[L]
        if len(set(cyc)) != len(cyc):
            final_ok = False
            break
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                final_ok = False
                break
        if not final_ok:
            break
    
    if not final_ok:
        # Reset to original
        edge_set = set()
        adj = {i: set() for i in range(N)}
        for a, b in edges:
            u, v = min(a, b), max(a, b)
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        for L in range(3, N + 1):
            if L == N:
                cycles[L] = list(range(N))
            elif L % 2 == 1:
                m = (L - 1) // 2
                cycles[L] = list(range(0, 2 * m + 1))
            else:
                m = L // 2
                if m == 1:
                    continue
                cycles[L] = [0] + list(range(2, 2 * m + 1))
    
    final_edges = list(edge_set)
    return {'edges': final_edges, 'cycles': [cycles[L] for L in range(3, N + 1)]}
