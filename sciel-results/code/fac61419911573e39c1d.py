"""Rewritten program: construct a triangle-free graph on 12 vertices with min degree >= 3,
minimizing distinct simple cycle lengths. Uses a single connected 3-regular bipartite graph."""

def construct():
    # Build a 3-regular bipartite graph on 12 vertices: parts A={0..5}, B={6..11}
    # Use a known 3-regular bipartite graph structure (e.g., based on Z6 x Z2 or custom)
    # This ensures all cycles are even (bipartite) and triangle-free automatically.
    
    # Construct using a circulant-like bipartite graph:
    # A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11}
    # Each a_i connects to b_{i}, b_{i+1}, b_{i+2} mod 6
    edges = set()
    for i in range(6):
        for j in range(3):
            a = i
            b = 6 + (i + j) % 6
            edges.add((min(a, b), max(a, b)))
    
    # Verify: 12 vertices, 18 edges (12*3/2 = 18), bipartite, 3-regular
    # All cycles are even length >= 4 (girth 4 for bipartite 3-reg on 12 verts typically)
    
    return [list(e) for e in sorted(edges)]
