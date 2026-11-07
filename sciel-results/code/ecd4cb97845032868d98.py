"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Convert to lists for shuffling
    adj_lists = {i: list(s) for i, s in adj.items()}
    
    # Set of all edges for availability tracking
    all_edges = set()
    for u, v in edges:
        if u > v: u, v = v, u
        all_edges.add((u, v))
        
    start_time = time.time()
    deadline = start_time + 55 # 55s limit to be safe
    
    def get_edge_key(u, v):
        if u > v: u, v = v, u
        return (u, v)
        
    def find_cycle(available_edges, adj_list):
        # Returns a list of vertices forming a Hamiltonian cycle, or None
        # We fix the start vertex to 0 to break rotational symmetry
        # We iterate neighbors of 0 in the order provided by adj_list
        
        # DFS
        path = [0]
        used_edges = set()
        
        def dfs(current, path, used_edges):
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if last connects to first
                key = get_edge_key(current, path[0])
                if key in available_edges and key not in used_edges:
                    return path[:]
                return None
            
            # Get neighbors
            for neighbor in adj_list[current]:
                if neighbor in path:
                    continue
                key = get_edge_key(current, neighbor)
                if key not in available_edges or key in used_edges:
                    continue
                
                # Pruning: If a vertex (other than start) has only 1 unused edge left, and we haven't used it,
                # and it's not the current vertex, we might be in trouble?
                # Simple degree check:
                # For any unvisited vertex v, if the number of available edges incident to v is < 2, fail.
                
                valid = True
                for v in range(n):
                    if v in path:
                        continue
                    # Count available edges incident to v
                    cnt = 0
                    for u in adj[v]:
                        k = get_edge_key(v, u)
                        if k in available_edges and k not in used_edges:
                            cnt += 1
                    if cnt < 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                used_edges.add(key)
                path.append(neighbor)
                res = dfs(neighbor, path, used_edges)
                if res is not None:
                    return res
                path.pop()
                used_edges.remove(key)
                
            return None
        
        # Try different starting neighbors for 0?
        # The DFS above starts with 0 and tries all neighbors.
        # This will find the first Hamiltonian cycle it encounters.
        return dfs(0, [0], set())
        
    # We need to find 3 disjoint cycles.
    # Backtracking on the set of cycles.
    
    def find_k_cycles(k, available_edges, adj_list):
        if time.time() > deadline:
            return None
            
        if k == 0:
            return []
            
        # Find a cycle
        cycle = find_cycle(available_edges, adj_list)
        if cycle is None:
            return None
            
        # Remove edges of cycle
        removed_edges = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            key = get_edge_key(u, v)
            if key in available_edges:
                available_edges.remove(key)
                removed_edges.append(key)
            else:
                # Should not happen if find_cycle was correct
                # Restore and fail
                for k in removed_edges:
                    available_edges.add(k)
                return None
                
        res = find_k_cycles(k-1, available_edges, adj_list)
        if res is not None:
            return [cycle] + res
            
        # Restore edges and try another cycle?
        # For simplicity and speed, if this path fails, we return None.
        # The caller can retry with randomization.
        for k in removed_edges:
            available_edges.add(k)
        return None

    # Retry mechanism with randomization
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        # Shuffle neighbors to explore different cycles
        for i in range(n):
            random.shuffle(adj_lists[i])
            
        current_edges = set(all_edges)
        result = find_k_cycles(3, current_edges, adj_lists)
        if result is not None:
            # Validate result
            # Check types
            final_cycles = []
            for c in result:
                final_cycles.append([int(v) for v in c])
            return final_cycles
            
    # Fallback: return the seed result if no 3-cycle packing found (should not happen)
    return [list(range(n))]

# EVOLVE-BLOCK-END
