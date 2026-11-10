import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list and edge set
    remaining_edges = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        if u > v: u, v = v, u
        remaining_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    cycles_found = []
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    def check_connectivity(visited, path):
        # Check if the unvisited vertices form a connected subgraph 
        # (or are connected to the ends of the path appropriately)
        # For a Hamiltonian path/cycle, the unvisited vertices must be connected 
        # to each other and to the current end, and eventually to the start.
        # Simple check: Are all unvisited vertices connected to each other?
        # If not, and there is more than 1 unvisited vertex, it's a dead end 
        # unless they can be connected via the path ends.
        # But the path is linear so far. The unvisited part must be a single chain 
        # that can be inserted.
        # Stronger: The graph induced by unvisited vertices must be connected.
        unvisited = [v for v in range(n) if v not in visited]
        if len(unvisited) <= 1:
            return True
        
        # BFS/DFS from the first unvisited vertex
        start_unvisited = unvisited[0]
        visited_set = set([start_unvisited])
        queue = [start_unvisited]
        
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor in visited_set:
                    continue
                if neighbor in visited: # Skip visited vertices (path nodes)
                    continue
                # Only traverse unvisited vertices
                if neighbor in unvisited:
                    visited_set.add(neighbor)
                    queue.append(neighbor)
        
        # If we didn't visit all unvisited vertices, they are disconnected
        if len(visited_set) != len(unvisited):
            return False
            
        return True

    def find_hamiltonian_cycle(start_vertex, heuristic_type):
        # heuristic_type: 0 = lowest degree, 1 = highest degree, 2 = index
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                edge_key = (0, last) if 0 < last else (last, 0)
                if edge_key in remaining_edges:
                    return path[:]
                return None
            
            curr = path[-1]
            neighbors = list(adj[curr])
            
            # Filter valid neighbors
            valid_neighbors = [nxt for nxt in neighbors if nxt not in visited and ((curr, nxt) in remaining_edges or (nxt, curr) in remaining_edges)]
            
            if not valid_neighbors:
                return None

            # Apply heuristic
            if heuristic_type == 0:
                valid_neighbors.sort(key=lambda x: len(adj[x]))
            elif heuristic_type == 1:
                valid_neighbors.sort(key=lambda x: -len(adj[x]))
            else:
                valid_neighbors.sort()
                
            for nxt in valid_neighbors:
                path.append(nxt)
                visited.add(nxt)
                
                # Pruning: Connectivity of unvisited
                if len(visited) < n:
                    if not check_connectivity(visited, path):
                        path.pop()
                        visited.remove(nxt)
                        continue
                        
                res = backtrack()
                if res is not None:
                    return res
                
                path.pop()
                visited.remove(nxt)
            
            return None
        
        return backtrack()

    # Try to find 3 cycles
    # We use a deterministic set of heuristics and start vertices to ensure reliability
    # Start vertices: 0 is standard. If it fails, try others.
    # Heuristics: 0 (Warnsdorff), 1 (Anti-Warnsdorff), 2 (Index)
    
    attempts = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    
    for start_v, heur in attempts:
        if time.time() > deadline:
            break
        if len(cycles_found) >= 3:
            break
            
        # Reset state for this attempt? No, we need to remove edges from previous cycles.
        # The state `remaining_edges` and `adj` are mutated.
        # We must be careful: if a cycle is found, we commit it.
        # If no cycle is found, we haven't changed state.
        
        cycle = find_hamiltonian_cycle(start_v, heur)
        
        if cycle is not None:
            # Commit the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge_key = (u, v) if u < v else (v, u)
                if edge_key in remaining_edges:
                    remaining_edges.remove(edge_key)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    # Error state, should not happen
                    break
            
            cycles_found.append(cycle)
            
            if len(cycles_found) >= 3:
                break
    if not cycles_found:
        # Fallback: If we failed completely, return an empty list or a known valid 1-cycle?
        # The task requires nonempty list.
        # If we can't find any, we fail. But we should find at least 1.
        # Let's try one more time with no pruning just to get *something*?
        # Or return a simple cycle if the graph is complete? It's not.
        # Given the graph structure, 3 cycles exist.
        # If we fail, it's a timeout or bug.
        return []

    # Ensure types are correct
    result = []
    for c in cycles_found:
        result.append([int(v) for v in c])
        
    return result
# EVOLVE-BLOCK-END
