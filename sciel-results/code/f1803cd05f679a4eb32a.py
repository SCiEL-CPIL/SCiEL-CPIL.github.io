"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build edge index and adjacency
    edge_set = set()
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        edge_index[key] = i
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def find_hc_robust(adj_list, start_vertices=None, max_attempts=10):
        """Find Hamiltonian cycle with multiple start vertices and randomized neighbor order."""
        if start_vertices is None:
            start_vertices = list(range(n))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                return None
            
            start = start_vertices[attempt % len(start_vertices)]
            path = [start]
            visited = {start}
            
            def dfs(u):
                if time.time() > deadline - 3:
                    return None
                
                if len(path) == n:
                    if start in adj_list[u]:
                        return list(path)
                    return None
                
                unvisited_neighbors = [v for v in adj_list[u] if v not in visited]
                if not unvisited_neighbors:
                    return None
                
                # Warnsdorff heuristic with randomization
                scored = []
                for v in unvisited_neighbors:
                    count = sum(1 for w in adj_list[v] if w not in visited and w != u)
                    # Add small random perturbation to break ties
                    scored.append((count + random.random() * 0.1, v))
                
                scored.sort()
                
                for _, v in scored:
                    path.append(v)
                    visited.add(v)
                    
                    res = dfs(v)
                    if res is not None:
                        return res
                    
                    path.pop()
                    visited.remove(v)
                
                return None
            
            result = dfs(start)
            if result is not None:
                return result
        
        return None
    
    def verify_three_cycles(cycles):
        """Verify that three cycles are valid, edge-disjoint, and use all vertices."""
        if len(cycles) != 3:
            return False
        
        all_vertices = set()
        all_edges_used = set()
        
        for cycle in cycles:
            if len(cycle) < 3 or len(set(cycle)) != len(cycle):
                return False
            all_vertices.update(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in all_edges_used:
                    return False
                if e not in edge_set:
                    return False
                all_edges_used.add(e)
        
        if all_vertices != set(range(n)):
            return False
        
        return True
    
    def find_three_cycles():
        """Find 3 edge-disjoint Hamiltonian cycles."""
        # Try multiple strategies
        for strategy in range(200):
            if time.time() > deadline - 5:
                break
            
            # Strategy: Find first HC, remove edges, find second HC, remove edges, check third
            adj_cur = {u: list(v) for u, v in adj.items()}
            
            # Randomize neighbor order for diversity
            for u in adj_cur:
                random.shuffle(adj_cur[u])
            
            hc1 = find_hc_robust(adj_cur, max_attempts=5)
            if hc1 is None:
                continue
            
            # Remove hc1 edges
            used_edges = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                used_edges.add((min(u, v), max(u, v)))
            
            # Build remaining graph
            adj_rem = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_rem[u].append(v)
                    adj_rem[v].append(u)
            
            # Check all vertices have degree 4
            possible = True
            for i in range(n):
                if len(adj_rem[i]) != 4:
                    possible = False
                    break
            if not possible:
                continue
            
            # Find second cycle in remaining 4-regular graph
            hc2 = find_hc_robust(adj_rem, max_attempts=5)
            if hc2 is None:
                continue
            
            # Remove hc2 edges
            for i in range(len(hc2)):
                u = hc2[i]
                v = hc2[(i + 1) % len(hc2)]
                used_edges.add((min(u, v), max(u, v)))
            
            # Build final remaining graph
            adj_final = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_final[u].append(v)
                    adj_final[v].append(u)
            
            # Check all vertices have degree 2
            possible = True
            for i in range(n):
                if len(adj_final[i]) != 2:
                    possible = False
                    break
            if not possible:
                continue
            
            # The remaining graph is 2-regular, so it's a union of cycles
            # Check if it's a single Hamiltonian cycle
            visited_final = set()
            cycle = [0]
            visited_final.add(0)
            current = 0
            
            while True:
                nexts = [v for v in adj_final[current] if v not in visited_final]
                if not nexts:
                    if 0 in adj_final[current] and len(cycle) == n:
                        break
                    else:
                        break
                current = nexts[0]
                cycle.append(current)
                visited_final.add(current)
                if len(cycle) > n:
                    break
            
            if len(cycle) == n and 0 in adj_final[cycle[-1]]:
                # Verify
                if verify_three_cycles([hc1, hc2, cycle]):
                    return [list(hc1), list(hc2), list(cycle)]
        
        return None
    
    def find_two_cycles():
        """Find 2 edge-disjoint Hamiltonian cycles."""
        for strategy in range(200):
            if time.time() > deadline - 3:
                break
            
            adj_cur = {u: list(v) for u, v in adj.items()}
            for u in adj_cur:
                random.shuffle(adj_cur[u])
            
            hc1 = find_hc_robust(adj_cur, max_attempts=5)
            if hc1 is None:
                continue
            
            used_edges = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                used_edges.add((min(u, v), max(u, v)))
            
            adj_rem = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_rem[u].append(v)
                    adj_rem[v].append(u)
            
            hc2 = find_hc_robust(adj_rem, max_attempts=5)
            if hc2 is None:
                continue
            
            # Verify edge disjointness
            all_edges = set()
            for c in [hc1, hc2]:
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i + 1) % len(c)]
                    e = (min(u, v), max(u, v))
                    if e in all_edges:
                        break
                    all_edges.add(e)
                else:
                    continue
                break
            else:
                return [list(hc1), list(hc2)]
        
        return None
    
    def find_one_cycle():
        """Find one Hamiltonian cycle."""
        adj_cur = {u: list(v) for u, v in adj.items()}
        for u in adj_cur:
            random.shuffle(adj_cur[u])
        
        hc = find_hc_robust(adj_cur, max_attempts=10)
        if hc is not None:
            return [hc]
        
        # Fallback: find any cycle of length >= 3
        for u, v in edges:
            for w in adj[u]:
                if w != v and (v, w) in edge_set or (w, v) in edge_set:
                    if u < v < w:
                        return [[int(u), int(v), int(w)]]
        
        return None
    
    # Try to find 3 cycles first
    result = find_three_cycles()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Fallback to 2 cycles
    result = find_two_cycles()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Fallback to 1 cycle
    result = find_one_cycle()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Last resort
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
