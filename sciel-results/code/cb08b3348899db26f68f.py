import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    This is the maximum n for the 'disjoint union of cliques' family.
    
    Proof of validity:
    1. C5-free: Each component is a K4. K4 is a complete graph on 4 vertices.
       It contains cycles of length 3, but no cycle of length 5 (requires 5 vertices).
       Since the graph is a disjoint union, no cycle can span multiple components.
    2. No independent set of size 9: The independence number of a disjoint union
       is the sum of the independence numbers of its components.
       alpha(K4) = 1. With 8 components, alpha(G) = 8.
       8 < 9, so the condition is satisfied.
    
    Maximality in family:
    - Max clique size is 4 (K5 contains C5).
    - Max number of components is 8 (to keep alpha < 9).
    - Max n = 8 * 4 = 32.
    """
    n = 32
    edges = []
    
    # Generate edges for 8 disjoint K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
    # EVOLVE-BLOCK-END
