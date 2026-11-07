"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
    
    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def is_triangle_free(adj, u, v):
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return False
        return True
    
    def can_add_edge(adj, u, v):
        if (min(u, v), max(u, v)) in edge_set:
            return False
        if abs(u - v) < 3:
            return False
        return is_triangle_free(adj, u, v)
    
    added_edges = []
    start_time = time.time()
    deadline = start_time + 80
    
    while True:
        if time.time() > deadline:
            break
        current_diam = bfs_diameter(adj)
        if current_diam <= 4:
            break
        
        best_edge = None
        best_diam = current_diam
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    d = bfs_diameter(adj)
                    adj[u][v] = False
                    adj[v][u] = False
                    if d < best_diam:
                        best_diam = d
                        best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        added_edges.append([u, v])
    
    final_diam = bfs_diameter(adj)
    if final_diam <= 4:
        return added_edges
    else:
        return added_edges

# EVOLVE-BLOCK-END
