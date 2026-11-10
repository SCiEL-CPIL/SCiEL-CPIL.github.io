import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sorted lists for deterministic ordering
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        adj[i].sort()
    
    degrees = [len(adj[i]) for i in range(n)]
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def _find_hc_fast(adj_list, used_edges, deadline, start=0):
        """Find Hamiltonian cycle with aggressive pruning."""
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        # Precompute which vertices still need to be visited
        remaining = n - 1
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                ek = edge_key(last, start)
                if ek not in used_edges and start in adj_list[last]:
                    return list(path)
                return None
            
            last = path[-1]
            
            # Get candidates
            candidates = []
            for nb in adj_list[last]:
                if not visited[nb]:
                    ek = edge_key(last, nb)
                    if ek not in used_edges:
                        # Pruning: check if nb has enough available edges to connect to remaining vertices
                        avail_count = 0
                        for x in adj_list[nb]:
                            if not visited[x] and x != start:
                                if edge_key(nb, x) not in used_edges:
                                    avail_count += 1
                        if len(path) == n - 1:
                            if start in adj_list[nb] and edge_key(nb, start) not in used_edges:
                                avail_count += 1
                        # Need at least 1 forward edge (or closing edge if last step)
                        if len(path) < n - 1 and avail_count < 1:
                            continue
                        if len(path) == n - 1 and avail_count < 1:
                            continue
                        candidates.append((avail_count, degrees[nb], nb))
            
            candidates.sort()
            
            for _, _, nb in candidates:
                path.append(nb)
                visited[nb] = True
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False
            
            return None
        
        return backtrack()
    
    def _find_hc_with_order(adj_list, used_edges, deadline, start=0, order=None):
        """Find HC trying vertices in a specific order for diversity."""
        if order is None:
            return _find_hc_fast(adj_list, used_edges, deadline, start)
        
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                ek = edge_key(last, start)
                if ek not in used_edges and start in adj_list[last]:
                    return list(path)
                return None
            
            last = path[-1]
            
            candidates = []
            for nb in adj_list[last]:
                if not visited[nb]:
                    ek = edge_key(last, nb)
                    if ek not in used_edges:
                        avail_count = 0
                        for x in adj_list[nb]:
                            if not visited[x] and x != start:
                                if edge_key(nb, x) not in used_edges:
                                    avail_count += 1
                        if len(path) == n - 1:
                            if start in adj_list[nb] and edge_key(nb, start) not in used_edges:
                                avail_count += 1
                        if len(path) < n - 1 and avail_count < 1:
                            continue
                        if len(path) == n - 1 and avail_count < 1:
                            continue
                        candidates.append((avail_count, degrees[nb], nb))
            
            candidates.sort()
            
            for _, _, nb in candidates:
                path.append(nb)
                visited[nb] = True
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False
            
            return None
        
        return backtrack()
    
    # Main search
    deadline = time.time() + 85
    
    best_result = []
    random.seed(42)
    
    # Strategy 1: Try to find 3 Hamiltonian cycles
    # First, find one HC
    used_edges = set()
    cycle1 = _find_hc_fast(adj, used_edges, deadline, start=0)
    if cycle1:
        for i in range(len(cycle1)):
            u = cycle1[i]
            v = cycle1[(i + 1) % len(cycle1)]
            used_edges.add(edge_key(u, v))
        best_result = [cycle1]
        
        # Try to find second HC
        cycle2 = _find_hc_fast(adj, used_edges, deadline, start=0)
        if cycle2:
            for i in range(len(cycle2)):
                u = cycle2[i]
                v = cycle2[(i + 1) % len(cycle2)]
                used_edges.add(edge_key(u, v))
            best_result.append(cycle2)
            
            # Try to find third HC
            cycle3 = _find_hc_fast(adj, used_edges, deadline, start=0)
            if cycle3:
                for i in range(len(cycle3)):
                    u = cycle3[i]
                    v = cycle3[(i + 1) % len(cycle3)]
                    used_edges.add(edge_key(u, v))
                best_result.append(cycle3)
    
    # If we didn't find 3, try random restarts with different start vertices
    if len(best_result) < 3:
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            used_edges = set()
            cycles = []
            
            # Try different start vertices in random order
            starts = list(range(n))
            random.shuffle(starts)
            
            for start in starts[:5]:  # Try up to 5 different starts
                if time.time() > deadline:
                    break
                if len(cycles) >= 3:
                    break
                
                cycle = _find_hc_fast(adj, used_edges, deadline, start=start)
                if cycle:
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add(edge_key(u, v))
            
            if len(cycles) > len(best_result):
                best_result = cycles
                if len(best_result) >= 3:
                    break
    
    # Ensure types are correct
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    # Fallback
    if not result:
        used_edges = set()
        cycle = _find_hc_fast(adj, used_edges, time.time() + 5, start=0)
        if cycle:
            result = [[int(v) for v in cycle]]
    
    return result
# EVOLVE-BLOCK-END
