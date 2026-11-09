"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Hub-and-spoke: connect vertex 12 to all even vertices (except 12 itself)
    # Vertex 12 is even. Even vertices: 0,2,4,6,8,10,12,14,16,18,20,22
    # Odd vertices: 1,3,5,7,9,11,13,15,17,19,21,23
    # 
    # Key insight: The path is bipartite with even/odd partition.
    # If we add edges only between even and odd vertices, the graph stays bipartite (triangle-free).
    # 
    # Strategy: Use vertex 12 (center) as a hub connected to all odd vertices.
    # This creates short paths through the hub.
    
    # Let's try: connect 12 to all odd vertices
    added = []
    for v in range(1, n, 2):  # odd vertices: 1,3,5,...,23
        if (12, v) not in base_edges and (v, 12) not in base_edges:
            added.append((12, v))
    
    # Verify: check triangle-free and diameter
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w in adj[u] and w != v:
                            return False
        return True
    
    # Compute diameter via BFS
    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    if not is_triangle_free(adj):
        # Fallback: use odd-difference edges (seed solution)
        added = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
        return [[int(u), int(v)] for u, v in added]
    
    diam = compute_diameter(adj)
    if diam > 4:
        # Need more edges. Try adding more hub connections or a second hub.
        # Try vertex 6 as second hub for left side
        added2 = list(added)
        adj2 = [set(s) for s in adj]
        
        # Add edges from 6 to odd vertices on the left (1,3,5) that aren't already connected
        for v in [1, 3, 5]:
            if (6, v) not in base_edges:
                # Check triangle-free
                common = adj2[6] & adj2[v]
                if not common:
                    adj2[6].add(v)
                    adj2[v].add(6)
                    added2.append((6, v))
        
        # Check if this helps
        if is_triangle_free(adj2) and compute_diameter(adj2) <= 4:
            added = added2
            adj = adj2
        else:
            # Fallback to seed
            added = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
            return [[int(u), int(v)] for u, v in added]
    
    return [[int(u), int(v)] for u, v in added]

# EVOLVE-BLOCK-END
