import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Start with the Groetzsch graph (11 vertices, 20 edges)
    # Vertices 0-4: C5
    # Vertices 5-9: Mycielski layer
    # Vertex 10: Mycielski top
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    for u, w in [(i, (i + 1) % 5) for i in range(5)]:
        edges.append([u, 5 + w])
        edges.append([w, 5 + u])
    for u in range(5):
        edges.append([5 + u, 10])
    
    # Vertices 11-15 are initially isolated
    # We will try to add edges involving 11-15 and between 0-10
    
    # Helper to check if adding edge (u, v) creates a triangle
    def creates_triangle(u, v, edge_set):
        for w in range(n):
            if w != u and w != v:
                if (min(u, w), max(u, w)) in edge_set and (min(v, w), max(v, w)) in edge_set:
                    return True
        return False

    # Helper to check if graph is 3-colorable
    def is_3_colorable(edge_list):
        # Simple backtracking
        color = [-1] * n
        adj = [[] for _ in range(n)]
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
            
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(color[neighbor] != c for neighbor in adj[v]):
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)

    # Convert edges to set for fast lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # List of potential edges to add
    potential_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                potential_edges.append((i, j))
    
    random.shuffle(potential_edges)
    
    best_edges = edges[:]
    best_m = len(edges)
    
    # Local search
    for u, v in potential_edges:
        if time.time() > deadline:
            break
        if creates_triangle(u, v, edge_set):
            continue
        # Try adding edge
        new_edges = edges + [[u, v]]
        new_edge_set = edge_set | {(min(u, v), max(u, v))}
        if is_3_colorable(new_edges):
            continue
        # If not 3-colorable, accept the edge
        edges = new_edges
        edge_set = new_edge_set
        best_edges = edges[:]
        best_m = len(edges)
        
        # Optionally, try to add more edges
        # But for now, just iterate through potential edges
    
    # If we didn't reach 32, try a few random restarts or swaps
    # But given time, return best found
    return best_edges
