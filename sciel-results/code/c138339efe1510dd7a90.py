"""
Constructs a C5-free graph with no independent set of size 9.
Uses a 4-partite construction based on K4 components to maximize n.

Approach:
- Use 8 disjoint K4 components (32 vertices total).
- K4 is C5-free (complete graph on 4 vertices has no cycles of length 5).
- Independence number of K4 is 1.
- Total independence number = 8 (one from each K4).
- No independent set of size 9 exists.
- This gives n=32, which is the best known lower bound for R(C5, K9).

The construction is verified:
1. No 5-cycles: Each component is K4, which has no 5-cycles.
2. No independent 9-set: Maximum independent set has size 8.
"""

def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    
    Returns:
        dict: {'n': 32, 'edges': [[u, v], ...]}
    """
    n = 32
    edges = []
    
    # Create 8 disjoint K4 components
    for b in range(8):
        base = 4 * b
        # Add all edges within this K4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
