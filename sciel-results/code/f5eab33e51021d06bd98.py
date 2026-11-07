"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _diameter2_and_essential(n, edges):
    """Check if graph on n vertices with given edges has diameter exactly 2
    and every edge is essential (removing it increases diameter > 2 or disconnects)."""
    if not edges:
        return False
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter is exactly 2 (connected and no pair at distance > 2)
    # BFS from each vertex
    def bfs_dist(start):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for i in range(len(queue)):
            u = queue[i]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist
    
    # Check diameter <= 2
    for s in range(n):
        d = bfs_dist(s)
        if any(dist == -1 for dist in d):
            return False  # disconnected
        if any(dist > 2 for dist in d):
            return False  # diameter > 2
    
    # Now check each edge is essential
    for idx, (u, v) in enumerate(edges):
        # Remove edge (u, v)
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Check if still diameter <= 2
        ok = True
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            for i in range(len(queue)):
                node = queue[i]
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            for d in dist:
                if d == -1 or d > 2:
                    ok = False
                    break
            if not ok:
                break
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        if ok:
            return False  # edge not essential
    
    return True

def construct():
    n = 16
    deadline = time.time() + 75  # safety margin before 90s
    
    # K_{8,8} is the unique extremal triangle-free graph on 16 vertices with 64 edges.
    # It has diameter 2 and every edge is essential.
    # Vertices 0-7 in one partition, 8-15 in the other.
    edges = []
    for i in range(8):
        for j in range(8):
            edges.append([i, j + 8])
    
    # Verify K_{8,8} satisfies all constraints
    if _diameter2_and_essential(n, edges):
        return edges
    
    # Fallback: if K_{8,8} somehow fails (shouldn't happen), try to find best
    # For now, just return K_{8,8} as it's theoretically optimal
    return edges

# EVOLVE-BLOCK-END
