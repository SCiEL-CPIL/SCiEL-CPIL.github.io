import random
import itertools

def construct():
    n = 20
    # Initialize with complete graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, random.randint(0, 19)])
    
    # Map edge to index
    edge_to_idx = {(u, v): i for i, (u, v, _) in enumerate(edges)}
    for (u, v, _) in edges:
        edge_to_idx[(v, u)] = edge_to_idx[(u, v)]
        
    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific ordering
    # There are 3 cycles per set of 4 vertices
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    # We just need to check if the 4 edges in the cycle have distinct colors.
    
    # To be efficient, we can just iterate through all combinations of 4 vertices
    # and check the 3 possible cycles.
    
    # However, removing edges changes the graph.
    # We will use a list of active edges.
    
    active = [True] * len(edges)
    
    # Helper to check if a cycle is valid
    def is_rainbow(cycle_edges):
        colors = [edges[i][2] for i in cycle_edges]
        return len(set(colors)) == 4

    # Helper to get cycle edges
    # Cycle (a,b,c,d) -> edges (a,b), (b,c), (c,d), (d,a)
    
    # Iterative removal
    # We need to be careful with performance.
    # Number of 4-cycles is ~14k.
    # We can loop until no changes.
    
    # To speed up, we can maintain a list of "bad" cycles.
    # But re-scanning all cycles is O(14k * 4) which is fast.
    
    # Let's try a simple loop:
    # While there exists a bad cycle:
    #   Find one.
    #   Remove a random edge from it.
    
    # Optimization: Only check cycles where all 4 edges are active.
    
    # Precompute vertex combinations
    vertices = list(range(n))
    quad_combos = list(itertools.combinations(vertices, 4))
    
    # For each quad, the 3 cycles are:
    # 0: (0,1,2,3)
    # 1: (0,1,3,2)
    # 2: (0,2,1,3)
    
    # We will store the edge indices for each cycle type for each quad
    # This might be too much memory/time to precompute fully, but 4845 quads * 3 is small.
    
    cycle_info = []
    for quad in quad_combos:
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx.get((a, b))
        e2 = edge_to_idx.get((b, c))
        e3 = edge_to_idx.get((c, d))
        e4 = edge_to_idx.get((d, a))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycle_info.append([e1, e2, e3, e4])
            
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx.get((a, b))
        e2 = edge_to_idx.get((b, d))
        e3 = edge_to_idx.get((d, c))
        e4 = edge_to_idx.get((c, a))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycle_info.append([e1, e2, e3, e4])
            
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx.get((a, c))
        e2 = edge_to_idx.get((c, b))
        e3 = edge_to_idx.get((b, d))
        e4 = edge_to_idx.get((d, a))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycle_info.append([e1, e2, e3, e4])
            
    # Now iterate
    # To avoid infinite loops or slow convergence, we can limit iterations.
    # But 14k cycles is small.
    
    # We will keep a list of active cycle indices
    # Actually, just scanning all cycles each time is O(14k).
    # If we do this 100 times, it's 1.4M operations. Very fast.
    
    for _ in range(1000): # Max iterations
        found_bad = False
        for cycle in cycle_info:
            if all(active[e] for e in cycle):
                if not is_rainbow(cycle):
                    # Remove a random edge from this cycle
                    idx_to_remove = random.choice(cycle)
                    active[idx_to_remove] = False
                    found_bad = True
                    break # Restart scan
        if not found_bad:
            break
            
    # Construct result
    result = []
    for i, (u, v, c) in enumerate(edges):
        if active[i]:
            result.append([u, v, c])
            
    return result
