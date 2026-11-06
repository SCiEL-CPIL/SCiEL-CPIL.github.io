"""Program that constructs a triangle-free graph with independence number < 10, maximizing n."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """Construct a triangle-free graph with alpha < 10, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # The Andrasfai graph And(9) on 26 vertices is the baseline
    # It's triangle-free with alpha = 9
    def andrasfai_26():
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 26 % 3 == 1]
        return {'n': n, 'edges': edges}
    
    best = andrasfai_26()
    
    # Check if we have time for a search
    if time.time() - start_time > deadline - 10:
        return best
    
    # Try to find a larger graph using random construction + local search
    # Strategy: Start with a random graph on n vertices, then:
    # 1. Remove edges to eliminate triangles
    # 2. Add edges to reduce independence number
    # 3. Verify constraints
    
    def is_triangle_free(adj, n):
        """Check if graph with adjacency matrix is triangle-free."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def max_independent_set_size(adj, n):
        """Exact maximum independent set size using branch and bound."""
        # Use a simple branch and bound
        best_size = [0]
        
        def dfs(vertices, current_size):
            if not vertices:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                return
            if current_size + len(vertices) <= best_size[0]:
                return
            
            v = vertices[0]
            rest = vertices[1:]
            
            # Branch 1: include v
            # Find neighbors of v in remaining vertices
            neighbors = [u for u in rest if adj[v][u]]
            non_neighbors = [u for u in rest if not adj[v][u]]
            dfs(non_neighbors, current_size + 1)
            
            # Branch 2: exclude v
            dfs(rest, current_size)
        
        dfs(list(range(n)), 0)
        return best_size[0]
    
    def verify_graph(n, edges):
        """Check if graph is triangle-free and alpha < 10."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        if not is_triangle_free(adj, n):
            return False
        
        if max_independent_set_size(adj, n) >= 10:
            return False
        
        return True
    
    # Try random constructions for n = 27, 28, 29, 30
    for target_n in range(27, 31):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple random attempts
        for attempt in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            random.seed(attempt * target_n + 42)
            
            # Start with a random graph
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.3:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Phase 1: Remove edges to make triangle-free
            for _ in range(100):
                # Find a triangle
                triangle = None
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i][j]:
                            for k in range(j+1, target_n):
                                if adj[i][k] and adj[j][k]:
                                    triangle = (i, j, k)
                                    break
                            if triangle:
                                break
                    if triangle:
                        break
                
                if triangle is None:
                    break
                
                # Remove a random edge from the triangle
                i, j, k = triangle
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    adj[i][j] = False
                    adj[j][i] = False
                elif edge_idx == 1:
                    adj[i][k] = False
                    adj[k][i] = False
                else:
                    adj[j][k] = False
                    adj[k][j] = False
            
            # Check if triangle-free
            if not is_triangle_free(adj, target_n):
                continue
            
            # Phase 2: Add edges to reduce independence number
            for _ in range(50):
                # Check current independence number
                alpha = max_independent_set_size(adj, target_n)
                if alpha < 10:
                    break
                
                # Try to add a random non-edge
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j or adj[i][j]:
                    continue
                
                # Check if adding this edge creates a triangle
                creates_triangle = False
                for k in range(target_n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        creates_triangle = True
                        break
                
                if not creates_triangle:
                    adj[i][j] = True
                    adj[j][i] = True
            
            # Final check
            if is_triangle_free(adj, target_n):
                alpha = max_independent_set_size(adj, target_n)
                if alpha < 10:
                    edges = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if adj[i][j]:
                                edges.append([i, j])
                    if target_n > best['n']:
                        best = {'n': target_n, 'edges': edges}
                    break
    
    return best
# EVOLVE-BLOCK-END
