import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Pre-compute initial adjacency and edge set
    init_adj = {i: set() for i in range(n)}
    init_edge_set = set()
    for u, v in edges:
        init_adj[u].add(v)
        init_adj[v].add(u)
        init_edge_set.add((min(u, v), max(u, v)))
        
    best_cycles = []
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try different starting vertices to break symmetry
    for start_node in range(n):
        if time.time() > deadline:
            break
            
        # Deep copy adj and edge_set for this attempt
        adj = {i: set(neigh) for i, neigh in init_adj.items()}
        edge_set = set(init_edge_set)
        
        cycles = []
        success = True
        
        for _ in range(3):
            if time.time() > deadline:
                success = False
                break
                
            cycle = find_ham_cycle(adj, n, start_node, start_time, deadline)
            if cycle is None:
                success = False
                break
                
            cycles.append(cycle)
            
            # Remove edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    edge_set.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                return best_cycles
                
    return best_cycles

def find_ham_cycle(adj, n, start_node, start_time, deadline):
    """Find a Hamiltonian cycle in the graph using DFS with heuristics."""
    visited = [False] * n
    path = [start_node]
    visited[start_node] = True
    
    def get_unvisited_neighbors(u):
        return [v for v in adj[u] if not visited[v]]
    
    def dfs(u):
        if time.time() > deadline:
            return False
            
        if len(path) == n:
            # Check if we can close the cycle
            if start_node in adj[u]:
                return True
            return False
            
        neighbors = get_unvisited_neighbors(u)
        
        if not neighbors:
            return False
            
        # Warnsdorff heuristic: prefer neighbors with fewer unvisited neighbors
        neighbors.sort(key=lambda v: len(get_unvisited_neighbors(v)))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            
            if dfs(v):
                return True
                
            path.pop()
            visited[v] = False
            
        return False
        
    if dfs(start_node):
        return list(path)
    return None

# EVOLVE-BLOCK-END
