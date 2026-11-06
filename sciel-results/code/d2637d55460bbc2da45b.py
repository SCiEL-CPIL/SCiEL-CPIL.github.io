# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _find_hamiltonian_cycle(adj, n):
    """
    Find a Hamiltonian cycle in the graph defined by adj.
    adj is a list of sets of neighbors.
    Returns a list of vertices [v0, v1, ..., v_{n-1}] or None.
    """
    if n == 0:
        return []
    
    # Check biconnectivity quickly? 
    # For n=24, we can just run DFS with pruning.
    
    # Sort neighbors by degree to help pruning (Warnsdorff)
    # We need to be careful: adj is static for this call.
    
    # Use bitmask for visited
    # Start at 0
    start = 0
    visited = 1 << start
    path = [start]
    
    def dfs(current, visited_mask, path):
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[current]:
                return path[:]
            return None
        
        # Pruning: Check if any unvisited vertex is isolated in the remaining graph
        # This is O(N^2) per step, which might be slow. 
        # Alternative: Just try to extend.
        
        # Get neighbors
        neighbors = list(adj[current])
        
        # Heuristic: Try neighbors with fewer options first
        # We can calculate the number of unvisited neighbors for each neighbor
        def heuristic_score(neighbor):
            count = 0
            for nb in adj[neighbor]:
                if not (visited_mask & (1 << nb)):
                    count += 1
            return count
            
        neighbors.sort(key=heuristic_score)
        
        for nb in neighbors:
            if visited_mask & (1 << nb):
                continue
            
            # Pruning: If we visit nb, does it isolate any other unvisited vertex?
            # Check if any unvisited vertex w (w != nb) has no unvisited neighbors other than nb?
            # Actually, if w only has nb as its unvisited neighbor, and we don't visit nb next, w is stuck.
            # But we ARE visiting nb. So w's options are reduced.
            # If w has degree 1 in the unvisited subgraph + current, and that neighbor is not nb, then w is stuck if we don't go to w?
            # Simpler check: After adding nb to visited, is the remaining unvisited graph connected to nb?
            
            new_visited = visited_mask | (1 << nb)
            
            # Quick check: Is there any unvisited vertex u such that all its neighbors are visited or u itself?
            # i.e., degree of u in (unvisited + {nb}) is 0?
            # If u is unvisited and u != nb, and all neighbors of u are in visited_mask, then u is isolated.
            # Wait, u can connect to nb. So u is not isolated if nb is in adj[u].
            
            # Let's do a connectivity check on the subgraph induced by unvisited vertices + nb.
            # This is expensive.
            
            # Let's just try the move.
            path.append(nb)
            result = dfs(nb, new_visited, path)
            if result:
                return result
            path.pop()
            
        return None
        
    return dfs(start, visited, path)

def construct():
    n = P['n']
    edge_order = P['edge_order']
    
    # Build edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[_edge_key(u, v)] = i + 1
        
    # Sort edges by position
    # We iterate through the edges in the order they appear.
    # We maintain the graph incrementally.
    
    adj = [set() for _ in range(n)]
    
    # We need to check after each edge addition?
    # There are 276 edges. Checking Hamiltonian cycle 276 times might be too slow if the check is expensive.
    # But the check is only run when the graph is biconnected.
    # Let's add a biconnectivity check.
    
    def is_biconnected(adj, n):
        if n < 3:
            return False
        # Tarjan's algorithm for biconnected components
        # If the number of biconnected components is 1, it's biconnected.
        # Or simply, if removing any vertex doesn't disconnect the graph.
        # For n=24, we can check: for each vertex v, remove v, check connectivity of remaining n-1 vertices.
        # If any removal disconnects, not biconnected.
        # This is O(N * (N+E)) which is fine.
        
        for v in range(n):
            # Remove v
            # Check connectivity of V \ {v}
            if n - 1 == 0:
                continue
            visited = [False] * n
            visited[v] = True # Mark as removed/visited to skip
            
            # BFS from an arbitrary neighbor of v? No, from any unvisited vertex.
            # Find first unvisited
            start = -1
            for u in range(n):
                if u != v and not visited[u]:
                    start = u
                    break
            if start == -1:
                continue # All removed? Impossible for n>1
                
            queue = [start]
            visited[start] = True
            
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if nb != v and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
                        
            # Check if all u != v are visited
            for u in range(n):
                if u != v and not visited[u]:
                    return False
                    
        return True

    # Iterate through edges
    for i, (u, v) in enumerate(edge_order):
        adj[u].add(v)
        adj[v].add(u)
        
        # Check biconnectivity
        if is_biconnected(adj, n):
            # Try to find Hamiltonian cycle
            cycle = _find_hamiltonian_cycle(adj, n)
            if cycle:
                return cycle
                
    # Fallback if not found (should not happen for complete graph eventually)
    return list(range(n))
# EVOLVE-BLOCK-END
