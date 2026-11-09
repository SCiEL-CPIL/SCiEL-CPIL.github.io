"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    best_edges = []
    best_count = 0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding (u, v) creates a C4
    # Graph represented as set of tuples (min, max)
    def is_c4_free(adj, edge_set, u, v):
        # adj[u] is set of neighbors
        # If we add (u, v), a C4 is formed if there exist x in N(u), w in N(v)
        # such that (x, w) is in edge_set and x != w.
        # Cycle: u - v - w - x - u
        # Edges: (u,v) [new], (v,w), (w,x), (x,u)
        # (v,w) exists => w in N(v)
        # (x,u) exists => x in N(u)
        # (w,x) exists => (min(w,x), max(w,x)) in edge_set
        # Also x != w to avoid triangle (u-v-w-u is 3-cycle, not 4)
        # Also x != v and w != u are guaranteed by simple graph and u,v not in N(u)/N(v) initially?
        # In a simple graph, u not in N(u).
        
        for x in adj[u]:
            for w in adj[v]:
                if x == w:
                    continue
                # Check if edge (x, w) exists
                key = (min(x, w), max(x, w))
                if key in edge_set:
                    return False
        return True

    # Greedy construction with random order
    max_attempts = 500  # Adjust based on speed
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE:
            break
            
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        current_edges = []
        
        for u, v in all_edges:
            if is_c4_free(adj, edge_set, u, v):
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_edges.append((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
        # Early stop if we reach a theoretical upper bound?
        # Reiman bound for n=20: ~48
        if best_count >= 48:
            break

    if not best_edges:
        # Fallback to star
        best_edges = [(0, v) for v in range(1, n)]

    # Construct the output format
    # Assign colors. Use all 20 colors if possible.
    # If we have >= 20 edges, assign colors 0..19 cyclically or randomly.
    # The task says "edge-coloured with 20 colours".
    # It doesn't strictly say "properly", but rainbow 4-cycle implies properness for edges in 4-cycles.
    # Since graph is C4-free, any coloring is valid.
    # To be safe, we can use a proper edge coloring if easy, or just assign colors.
    # Simplest: assign color i % 20 to the i-th edge.
    
    final_output = []
    for i, (u, v) in enumerate(best_edges):
        c = i % 20
        final_output.append([u, v, c])
        
    return final_output
# EVOLVE-BLOCK-END
