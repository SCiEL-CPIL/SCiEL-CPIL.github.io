import numpy as np

def construct():
    # Use a Complete Bipartite Graph K(10, 10)
    # Vertices 0..9 in U, 10..19 in V
    # Edges between u in U and v in V
    # Color c(u, v) = (u + 10 * (v - 10)) % 20
    # This ensures every 4-cycle is rainbow
    
    edges = []
    for u in range(10):
        for v in range(10, 20):
            # Map v to 0..9 for calculation
            v_idx = v - 10
            c = (u + 10 * v_idx) % 20
            edges.append([u, v, c])
            
    return edges
