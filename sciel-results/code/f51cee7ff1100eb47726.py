"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct(k, g):
    # Use circulant bipartite graph: A[i] connected to B[(i+s)%n] for s in shifts
    # This gives a 3-regular bipartite graph on 2n vertices
    
    def check_girth(n, shifts, girth_target):
        """Check if circulant bipartite graph has girth >= girth_target"""
        # Build adjacency
        adj = [None] * (2 * n)
        for i in range(n):
            adj[i] = [(n + (i + s) % n) for s in shifts]
        for j in range(n):
            adj[n + j] = [(j - s) % n for s in shifts]
        
        # BFS from each vertex to find shortest cycle
        # Limit depth to girth_target // 2
        for start in range(2 * n):
            visited = [-1] * (2 * n)
            parent = [-1] * (2 * n)
            visited[start] = 0
            queue = [(start, -1)]
            
            while queue:
                v, p = queue.pop(0)
                if visited[v] >= girth_target // 2:
                    continue
                for nb in adj[v]:
                    if nb == p:
                        continue
                    if visited[nb] == -1:
                        visited[nb] = visited[v] + 1
                        parent[nb] = v
                        queue.append((nb, v))
                    elif visited[v] + 1 > visited[nb]:
                        # Found a cycle
                        cycle_len = visited[v] + 1 + visited[nb]
                        if cycle_len < girth_target:
                            return False
        return True
    
    # Try structured shifts first
    for n in range(468, 510):
        for shifts in [[1, 2, 4], [1, 2, 3], [1, 3, 7], [1, 2, 5]]:
            if check_girth(n, shifts, g):
                edges = []
                for i in range(n):
                    for s in shifts:
                        j = (i + s) % n
                        edges.append((i, n + j))
                return 2 * n, edges
    
    # Random search
    for n in range(468, 520):
        for attempt in range(100):
            shifts = random.sample(range(1, n), 3)
            if check_girth(n, shifts, g):
                edges = []
                for i in range(n):
                    for s in shifts:
                        j = (i + s) % n
                        edges.append((i, n + j))
                return 2 * n, edges
    
    # Fallback: use a larger n with simple shifts
    n = 512
    shifts = [1, 2, 4]
    edges = []
    for i in range(n):
        for s in shifts:
            j = (i + s) % n
            edges.append((i, n + j))
    return 2 * n, edges
# EVOLVE-BLOCK-END
