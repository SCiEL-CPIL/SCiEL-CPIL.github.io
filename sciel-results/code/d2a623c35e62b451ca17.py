"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K4-free graph on 18 vertices minimizing the size of the largest
    triangle-free induced subgraph.
    
    Strategy: Use a random greedy algorithm with local search to find a K4-free
    graph where the largest triangle-free induced subgraph is as small as possible.
    We start with a known construction and then try to improve it.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n = 18
    
    def is_k4_free(adj):
        """Check if the graph represented by adjacency matrix is K4-free."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            for x in range(w+1, n):
                                if adj[u][x] and adj[v][x] and adj[w][x]:
                                    return False
        return True
    
    def max_triangle_free_induced(adj):
        """
        Find the size of the largest triangle-free induced subgraph using
        branch and bound.
        """
        best = 0
        
        # Try all subsets in decreasing order of size
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                # Check if this subset is triangle-free
                is_tf = True
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        for k in range(j+1, len(subset)):
                            u, v, w = subset[i], subset[j], subset[k]
                            if adj[u][v] and adj[v][w] and adj[u][w]:
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                
                if is_tf:
                    best = size
                    found = True
                    break
            
            if found:
                break
        
        return best
    
    def count_triangles(adj, subset):
        """Count triangles in the induced subgraph on subset."""
        count = 0
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                for k in range(j+1, len(subset)):
                    u, v, w = subset[i], subset[j], subset[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        count += 1
        return count
    
    # Start with the complete tripartite graph K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [[0]*n for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = 1
                    adj[b][a] = 1
    
    # Verify it's K4-free
    if not is_k4_free(adj):
        # Fallback to empty graph
        return []
    
    # Calculate initial score
    initial_score = max_triangle_free_induced(adj)
    
    # Local search: try to add edges to reduce the largest triangle-free subgraph
    # An edge (u,v) can be added if it doesn't create a K4
    # and it might reduce the size of the largest triangle-free induced subgraph
    
    best_adj = [row[:] for row in adj]
    best_score = initial_score
    
    # Try to add edges greedily
    added_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if u < v and adj[u][v] == 0:
                # Check if adding this edge keeps the graph K4-free
                can_add = True
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        # Check if there's a common neighbor x of u, v, w
                        for x in range(n):
                            if x != u and x != v and x != w:
                                if adj[u][x] and adj[v][x] and adj[w][x]:
                                    can_add = False
                                    break
                        if not can_add:
                            break
                
                if can_add:
                    # Try adding the edge
                    adj[u][v] = 1
                    adj[v][u] = 1
                    added_edges.add((u, v))
                    
                    new_score = max_triangle_free_induced(adj)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                    else:
                        # Revert
                        adj[u][v] = 0
                        adj[v][u] = 0
                        added_edges.discard((u, v))
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
