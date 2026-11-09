import time
import sys

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # 1. Initial Construction: Rim + 18 Chords (58 edges)
    # Rim: 40 edges
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
    # Chords: (0, k) for k = 3, 4, ..., 20
    # Each chord (0,k) creates cycles of length k and 41-k
    # k=3: lengths 3, 38
    # k=4: lengths 4, 37
    # ...
    # k=19: lengths 19, 22
    # k=20: lengths 20, 21
    chord_edges = [(0, k) for k in range(3, 21)]
    
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
    
    # Build adjacency list for fast lookup
    adj = {i: set() for i in range(N)}
    edge_set = set()
    for a, b in edges:
        u, v = min(a, b), max(a, b)
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    # 2. Initial Cycle Generation (Geometric)
    def get_initial_cycles(edge_set):
        cycles = {}
        for L in range(3, N + 1):
            if L == N:
                cyc = list(range(N))
            elif L <= 20:
                # Use chord (0, L): cycle is 0, L, L+1, ..., 39, 0? No.
                # Chord (0,k) gives cycle of length k: 0, k-1, k-2, ..., 1, 0
                # Wait, let me reconsider.
                # The two cycles from chord (0,k) are:
                # Cycle 1: 0 -> k -> k+1 -> ... -> 39 -> 0 (length 41-k)
                # Cycle 2: 0 -> 1 -> 2 -> ... -> k -> 0 (length k+1)
                # Actually, the rim goes 0-1-2-...-39-0.
                # Adding chord (0,k) creates two cycles:
                # - The "short" arc: 0-1-2-...-k-0 (length k+1)
                # - The "long" arc: 0-k-(k+1)-...-39-0 (length N-k+1 = 41-k)
                
                if L == 2:
                    cycles[L] = None
                    continue
                # For cycle of length L, we need k+1 = L or 41-k = L
                # So k = L-1 or k = 41-L
                k1 = L - 1
                k2 = 41 - L
                
                found = False
                # Try k1
                if 3 <= k1 <= 20:
                    # Cycle: 0, 1, 2, ..., k1, 0 (length k1+1 = L)
                    cyc = list(range(0, k1 + 1))
                    # Verify
                    valid = True
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i + 1) % len(cyc)]
                        if (min(u, v), max(u, v)) not in edge_set:
                            valid = False
                            break
                    if valid:
                        cycles[L] = cyc
                        found = True
                
                # Try k2
                if not found and 3 <= k2 <= 20:
                    # Cycle: 0, k2, k2+1, ..., 39, 0 (length 41-k2 = L)
                    cyc = [0] + list(range(k2, N))
                    # Verify
                    valid = True
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i + 1) % len(cyc)]
                        if (min(u, v), max(u, v)) not in edge_set:
                            valid = False
                            break
                    if valid:
                        cycles[L] = cyc
                        found = True
                
                if not found:
                    cycles[L] = None
            else:
                # L > 20, use k = 41-L
                k = 41 - L
                if 3 <= k <= 20:
                    # Cycle: 0, k, k+1, ..., 39, 0
                    cyc = [0] + list(range(k, N))
                    valid = True
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i + 1) % len(cyc)]
                        if (min(u, v), max(u, v)) not in edge_set:
                            valid = False
                            break
                    if valid:
                        cycles[L] = cyc
                    else:
                        cycles[L] = None
                else:
                    cycles[L] = None
        return cycles

    cycles = get_initial_cycles(edge_set)
    
    # Check if all cycles were found
    missing = [L for L in range(3, N + 1) if cycles[L] is None]
    if missing:
        # Fallback: try to find missing cycles via DFS
        for L in missing:
            if time.time() - t0 > DEADLINE_SECONDS - 5:
                break
            cyc = _find_cycle_dfs(L, adj, edge_set)
            if cyc:
                cycles[L] = cyc
    
    if any(c is None for c in cycles.values()):
        # Return what we have
        final_cycles = []
        for L in range(3, N + 1):
            if cycles[L] is not None:
                final_cycles.append(cycles[L])
            else:
                final_cycles.append([0, 1, 2])  # placeholder
        return {'edges': list(edge_set), 'cycles': final_cycles}

    # 3. Greedy Edge Removal
    # Sort edges to try removing chords first
    candidate_edges = list(edge_set)
    
    def edge_sort_key(e):
        u, v = e
        # Chords (involving 0) are less critical
        if u == 0 or v == 0:
            return (0, abs(u - v))
        else:
            return (1, 0)
            
    candidate_edges.sort(key=edge_sort_key)
    
    def _find_cycle_dfs(L, adj, edge_set, time_limit=0.3):
        """Find a simple cycle of length L using DFS with pruning."""
        if L < 3 or L > N:
            return None
        
        t_start = time.time()
        
        def dfs_from(start):
            visited = [False] * N
            path = []
            path_set = set()
            
            def dfs(u, depth):
                if time.time() - t_start > time_limit:
                    return None
                
                if depth == L - 1:
                    if start in adj[u]:
                        return path[:]
                    return None
                
                visited[u] = True
                path.append(u)
                path_set.add(u)
                
                neighbors = sorted(adj[u], key=lambda x: (x in path_set, abs(x - start)))
                for v in neighbors:
                    if v == start:
                        continue
                    if not visited[v]:
                        res = dfs(v, depth + 1)
                        if res is not None:
                            return res
                
                path.pop()
                path_set.remove(u)
                visited[u] = False
                return None
            
            return dfs(start, 0)
        
        # Try starting from 0 first
        res = dfs_from(0)
        if res is not None:
            return res
        
        # Try other starting nodes
        for s in range(1, N):
            if time.time() - t_start > time_limit:
                break
            res = dfs_from(s)
            if res is not None:
                return res
                
        return None

    for e in candidate_edges:
        if time.time() - t0 > DEADLINE_SECONDS - 5:
            break
            
        u, v = e
        if e not in edge_set:
            continue
            
        # Check which cycles use this edge
        critical_lengths = []
        for L in range(3, N + 1):
            cyc = cycles[L]
            if cyc is None:
                continue
            for i in range(len(cyc)):
                c_u = cyc[i]
                c_v = cyc[(i + 1) % len(cyc)]
                if (min(c_u, c_v), max(c_u, c_v)) == e:
                    critical_lengths.append(L)
                    break
        
        if not critical_lengths:
            # Edge not used in any current cycle - safe to remove
            edge_set.remove(e)
            adj[u].remove(v)
            adj[v].remove(u)
            continue
            
        # Edge is used in critical_lengths. Try to find replacement cycles.
        edge_set.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
        all_valid = True
        new_cycles = {}
        
        for L in range(3, N + 1):
            if L in critical_lengths:
                new_cyc = _find_cycle_dfs(L, adj, edge_set)
                if new_cyc is None:
                    all_valid = False
                    break
                new_cycles[L] = new_cyc
            else:
                new_cycles[L] = cycles[L]
        
        if all_valid:
            cycles = new_cycles
        else:
            # Restore edge
            edge_set.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
    # Final Validation
    final_edges = list(edge_set)
    final_cycles = [cycles[L] for L in range(3, N + 1)]
    
    # Verify final result
    ok = True
    for L, cyc in zip(range(3, N + 1), final_cycles):
        if not cyc or len(cyc) != L:
            ok = False
            break
        if len(set(cyc)) != L:
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
        # Fallback to initial 58-edge construction
        edges = rim_edges + chord_edges
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
        edge_set = set((min(a, b), max(a, b)) for a, b in edges)
        cycles = get_initial_cycles(edge_set)
        final_cycles = [cycles[L] for L in range(3, N + 1)]
        final_edges = edges

    return {'edges': final_edges, 'cycles': final_cycles}
