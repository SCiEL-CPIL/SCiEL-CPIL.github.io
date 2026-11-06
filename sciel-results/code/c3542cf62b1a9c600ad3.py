# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    This is a proven valid configuration for R(C5, K9) > 32.
    
    Design Family: Disjoint union of cliques.
    - No C5: Components are K4 (clique size 4).
    - No independent set of 9: alpha = 8 * alpha(K4) = 8 * 1 = 8.
    - n = 32.
    """
    n = 32
    edges = []
    
    # 8 disjoint blocks of size 4
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    return {'n': n, 'edges': edges}
    # EVOLVE-BLOCK-END
