"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 50  # Safety margin before 60s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    n = P['n']
    edges_list = P['edges']
    m = len(edges_list)
    
    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i
    
    # Adjacency with edge indices
    adj = [set() for _ in range(n)]
    adj_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges_list):
        adj[u].add(v)
        adj[v].add(u)
        adj_edges[u].append((v, idx))
        adj_edges[v].append((u, idx))
        
    for v in range(n):
        adj_edges[v].sort()
    
    max_possible_cycles = min(10, 6 // 2, m // n)  # 3
    
    def find_hamiltonian_cycles(k_target):
        used_edges = [False] * m
        cycles = []
        
        # Precompute full mask
        full_mask = (1 << n) - 1
        
        def check_connectivity(unvisited_mask, used_edges_ref):
            """Check if all unvisited vertices are connected via unused edges using bitmask BFS."""
            if unvisited_mask == 0:
                return True
            # Find first unvisited vertex
            start = (unvisited_mask & (-unvisited_mask)).bit_length() - 1
            visited = 1 << start
            stack = [start]
            while stack:
                v = stack.pop()
                for neighbor, eidx in adj_edges[v]:
                    bit = 1 << neighbor
                    if (unvisited_mask & bit) and not (visited & bit):
                        if not used_edges_ref[eidx]:
                            visited |= bit
                            stack.append(neighbor)
            return visited == unvisited_mask
        
        def dfs_cycle(start, path, visited_mask, used_edges_ref, depth, remaining_deg):
            if time.time() > deadline:
                return None
            
            if depth == n:
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in edge_index:
                    eidx = edge_index[key]
                    if not used_edges_ref[eidx]:
                        return list(path)
                return None
            
            last = path[-1]
            
            candidates = []
            for neighbor, eidx in adj_edges[last]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges_ref[eidx]:
                    continue
                if remaining_deg[neighbor] < 2:
                    continue
                candidates.append((remaining_deg[neighbor], neighbor, eidx))
            
            candidates.sort()
            
            for rem_deg, neighbor, eidx in candidates:
                visited_mask_new = visited_mask | (1 << neighbor)
                path.append(neighbor)
                used_edges_ref[eidx] = True
                remaining_deg[last] -= 1
                remaining_deg[neighbor] -= 1
                
                # Forward checking: connectivity of unvisited vertices
                if depth + 1 < n:
                    unvisited_mask = full_mask & ~visited_mask_new
                    if unvisited_mask and not check_connectivity(unvisited_mask, used_edges_ref):
                        # Undo and skip
                        remaining_deg[last] += 1
                        remaining_deg[neighbor] += 1
                        used_edges_ref[eidx] = False
                        path.pop()
                        continue
                
                result = dfs_cycle(start, path, visited_mask_new, used_edges_ref, depth + 1, remaining_deg)
                
                remaining_deg[last] += 1
                remaining_deg[neighbor] += 1
                used_edges_ref[eidx] = False
                path.pop()
                
                if result is not None:
                    return result
            
            return None
        
        for c in range(k_target):
            if time.time() > deadline - 5:
                break
            
            remaining_deg = [0] * n
            for v in range(n):
                for neighbor, eidx in adj_edges[v]:
                    if not used_edges[eidx]:
                        remaining_deg[v] += 1
            
            if remaining_deg[0] < 2:
                break
                
            found = dfs_cycle(0, [0], 1, used_edges, 1, remaining_deg)
            
            if found is None:
                break
            
            for i in range(n):
                u = found[i]
                v = found[(i + 1) % n]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges[eidx] = True
            
            cycles.append(found)
        
        return cycles

    best_cycles = find_hamiltonian_cycles(max_possible_cycles)
    
    if not best_cycles:
        # Fallback: find any triangle
        for i, (u, v) in enumerate(edges_list):
            if time.time() > deadline:
                break
            for w in adj[u] & adj[v]:
                if w != u and w != v:
                    return [[int(u), int(v), int(w)]]
    
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
