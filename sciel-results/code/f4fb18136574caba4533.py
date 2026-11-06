import time
import random
import sys

def construct():
    deadline = time.time() + 130
    n = 20
    K = 20
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # We will run the greedy algorithm multiple times with different random orderings
    # and keep the best result.
    
    # Number of attempts depends on time
    # Each attempt is very fast, so we can do many.
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Randomize edge order
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        # adj[u][v] = color
        adj = {i: {} for i in range(n)}
        current_edges = []
        
        for u, v in edges_to_try:
            if time.time() > deadline - 2:
                break
                
            # Find valid colors for (u, v)
            valid_colors = set(range(K))
            
            # Identify all 4-cycles that would be formed by adding (u, v)
            # Cycle: u - v - w - x - u
            # Edges: (u,v) [new], (v,w), (w,x), (x,u)
            # We need (v,w), (w,x), (x,u) to exist in the current graph.
            
            # Iterate neighbors of v
            # Note: adj[v] contains neighbors w such that (v,w) is an edge
            for w, c_vw in adj[v].items():
                # Iterate neighbors of u
                for x, c_xu in adj[u].items():
                    # Check if (w, x) is an edge
                    if w == x:
                        continue
                    if x in adj[w]:
                        c_wx = adj[w][x]
                        # The cycle is u-v-w-x-u
                        # Colors: c_uv (to be determined), c_vw, c_wx, c_xu
                        # We need c_uv, c_vw, c_wx, c_xu to be distinct.
                        # So c_uv must not be in {c_vw, c_wx, c_xu}
                        forbidden = {c_vw, c_wx, c_xu}
                        valid_colors -= forbidden
            
            if valid_colors:
                # Pick a color. 
                # Heuristic: Pick a color that appears least frequently in the graph?
                # Or just the first one.
                # For now, pick the smallest valid color to encourage consistency?
                # Actually, random might be better to avoid local minima.
                c = random.choice(list(valid_colors))
                
                # Add edge
                adj[u][v] = c
                adj[v][u] = c
                current_edges.append([u, v, c])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
            # Early stop if we reach a very high density?
            # Max possible is 190.
            if best_count >= 150:
                # Try a few more to see if we can go higher, but don't wait long
                if time.time() > deadline - 30:
                    break

    # If we didn't find anything, return empty (should not happen)
    if not best_edges:
        # Fallback: Return a small valid graph, e.g., a star
        edges = []
        for i in range(1, n):
            edges.append([0, i, i % K])
        return edges
        
    return best_edges
