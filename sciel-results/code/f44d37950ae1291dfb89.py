"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    # We want to maximize edges.
    # A valid coloring must have distinct colors on every 4-cycle.
    # This is a hard constraint satisfaction problem.
    # Greedy approach:
    # 1. Start with empty graph.
    # 2. Add edges one by one in a random order.
    # 3. For each edge (u, v), try all available colors.
    # 4. Accept the first color that does not create a non-rainbow 4-cycle.
    # 5. If no color works, skip the edge.
    # 6. Repeat with multiple random orders and restarts to find a dense solution.
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # To check if adding edge (u, v) with color c creates a bad 4-cycle:
    # We need to check all pairs of common neighbors (a, b) of u and v.
    # The cycle would be u-a-v-b-u.
    # The colors of edges (u,a), (a,v), (v,b), (b,u) must be distinct.
    # Note: (a,v) and (v,a) are the same edge, so we need color of (a,v) and (b,u).
    # Wait, the cycle is u-a-v-b-u. Edges: (u,a), (a,v), (v,b), (b,u).
    # We are adding (u,v) with color c.
    # The other 3 edges must exist and have distinct colors from each other and from c.
    # So for each pair of common neighbors a, b of u and v:
    #   Check if (u,a), (a,v), (v,b), (b,u) all exist.
    #   Check if colors are distinct: col(u,a) != col(a,v) != col(v,b) != col(b,u) != col(u,a)
    #   AND none of them is c.
    
    # Optimization:
    # Maintain adjacency list with colors: adj[u][v] = color
    # Maintain set of common neighbors? No, just iterate.
    
    def is_valid_addition(adj, u, v, c):
        # Get common neighbors of u and v
        # adj is a dict of dicts: adj[u][v] = c
        # Common neighbors are w such that adj[u].get(w) is not None and adj[v].get(w) is not None
        # But we need to check cycles u-w-v-x-u? No.
        # A 4-cycle involving edge (u,v) must be of the form u-v-w-x-u? No, (u,v) is an edge.
        # The cycle is u-v-w-x-u.
        # Edges: (u,v), (v,w), (w,x), (x,u).
        # We add (u,v) with color c.
        # We need to check all pairs (w, x) such that:
        # 1. (v,w) exists
        # 2. (w,x) exists
        # 3. (x,u) exists
        # 4. w != x, w != u, w != v, x != u, x != v
        # 5. Colors of (v,w), (w,x), (x,u) are distinct from each other and from c.
        
        # Iterate over neighbors w of v
        for w in adj.get(v, {}):
            if w == u: continue
            # Iterate over neighbors x of w
            for x in adj.get(w, {}):
                if x == u or x == v or x == w: continue
                # Check if (x, u) exists
                if x not in adj.get(u, {}): continue
                
                # Check colors
                c1 = adj[v][w]
                c2 = adj[w][x]
                c3 = adj[u][x]
                
                # Colors must be distinct from each other and from c
                if c1 == c2 or c1 == c3 or c2 == c3:
                    continue # Already a non-rainbow cycle? No, we assume existing graph is valid.
                             # So existing cycles are rainbow. We only need to check if new color c conflicts.
                             # But wait, if c1, c2, c3 are distinct, we just need c to be distinct from them.
                             # However, we must also ensure that the cycle u-v-w-x-u is indeed a 4-cycle.
                             # Yes, distinct vertices u, v, w, x.
                             
                if c in (c1, c2, c3):
                    return False
        return True

    # This check is O(deg(v) * deg(w)) which can be slow if graph is dense.
    # For n=20, max degree 19. 19*19 = 361 checks per edge addition.
    # 190 edges * 361 * 20 colors = ~1.4M operations per restart. Very fast.
    
    def greedy_attempt(edge_order, time_limit):
        adj = {}
        edges = []
        
        for u, v in edge_order:
            if time.time() > time_limit:
                break
            # Try colors 0..19
            found = False
            # Shuffle colors for randomness? Or just try 0..19.
            # Randomizing color order might help avoid local minima?
            # Let's try colors in random order.
            colors = list(range(20))
            random.shuffle(colors)
            for c in colors:
                if is_valid_addition(adj, u, v, c):
                    # Add edge
                    adj.setdefault(u, {})[v] = c
                    adj.setdefault(v, {})[u] = c
                    edges.append([u, v, c])
                    found = True
                    break
            # If not found, skip edge
        return edges

    # Multiple restarts
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        # Random edge order
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        # Run greedy
        edges = greedy_attempt(edge_order, DEADLINE)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        
        # If we found a very dense graph (e.g., > 150 edges), we might stop early?
        # No, keep trying to improve.
        
        # Heuristic: If we've done many attempts and not improving, break?
        # For now, just run until deadline.
        
        # Optimization: If best_count is 0, something is wrong.
        # But with 20 colors and n=20, we should be able to add many edges.
        
        # Let's add a small sleep if we are finishing too fast? No, just run more.
        
        # To diversify, we can also try different random seeds for color shuffling implicitly.
        
        # Check if we hit the deadline in the inner loop
        if time.time() > DEADLINE:
            break

    return best_edges

# EVOLVE-BLOCK-END
