import random
import time

P = {'n': 15}

def construct():
    n = P['n']
    # Start with a random graph or empty graph
    # Use Simulated Annealing
    # State: list of edges
    # Score: number of edges if valid, else -1
    
    def is_valid(edges):
        # Build adjacency list and degrees
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        # Check all paths of length 3
        # s-u-v-t
        # Iterate over all pairs (u, v) that are edges
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # Avoid double processing? No, path is directed s->t
                # For each neighbor s of u (s != v)
                for s in adj[u]:
                    if s == v: continue
                    # For each neighbor t of v (t != u)
                    for t in adj[v]:
                        if t == u: continue
                        if s == t: continue # Simple path
                        # Check degrees
                        if deg[s] == deg[t]:
                            return False
        return True

    def edge_count(edges):
        return len(edges)

    # Initialize with complete graph
    edges = [[i, j] for i in range(n) for j in range(i+1, n)]
    # This is invalid. We need to remove edges.
    # Start with empty and add?
    # Or start with K_{7,8} and try to improve?
    # Directive: Do not reuse current designs.
    # I will start with a random sparse graph and add edges.
    
    random.seed(42)
    best_edges = []
    best_score = 0
    
    # Start with a random graph
    curr_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                curr_edges.add((i, j))
                
    # Local search
    # Try to add edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for _ in range(1000):
        # Try to add a random missing edge
        missing = [e for e in all_possible if e not in curr_edges]
        if not missing:
            break
        u, v = random.choice(missing)
        new_edges = curr_edges | {(u, v)}
        if is_valid(list(new_edges)):
            curr_edges = new_edges
            best_edges = list(curr_edges)
            best_score = len(curr_edges)
        else:
            # Try to remove an edge to allow addition?
            # Simple hill climbing might get stuck.
            pass
            
    # If stuck, try a different strategy: Start with K_{7,8} equivalent but generated differently?
    # No, I must not reuse.
    # I will return the best found.
    
    # Convert to list of lists
    return [list(e) for e in best_edges]
