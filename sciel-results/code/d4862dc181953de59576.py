import time
import random
import sys

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, u, v, c):
        # Check all 4-cycles containing (u, v)
        # Cycle: u-v-w-x-u
        # Edges: (u,v) [new], (v,w), (w,x), (x,u) [existing]
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == -1: continue
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[w][x] == -1: continue
                if adj[x][u] == -1: continue
                # Cycle exists: u-v-w-x-u
                c1 = adj[v][w]
                c2 = adj[w][x]
                c3 = adj[x][u]
                # New colour is c
                # Colours must be distinct
                if len({c, c1, c2, c3}) < 4:
                    return False
        return True

    def try_construction(seed):
        random.seed(seed)
        adj = [[-1] * n for _ in range(n)]
        edges = []
        
        # Shuffle edge order
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        for u, v in edge_order:
            # Try to find a valid colour
            colours = list(range(20))
            random.shuffle(colours)
            added = False
            for c in colours:
                if is_valid_addition(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    added = True
                    break
            # If no colour works, skip edge
            if not added:
                continue
        
        return edges

    # Run multiple restarts
    start_time = time.time()
    seed = 0
    while time.time() < deadline:
        edges = try_construction(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        seed += 1
        
        # If we reach a very high density, we might break early?
        # Max possible is 190. 100+ is good.
        # But let's keep going to see if we can push higher.
        
    return best_edges

# EVOLVE-BLOCK-START
# The above code is the implementation. 
# I need to wrap it in the required format.
# EVOLVE-BLOCK-END
