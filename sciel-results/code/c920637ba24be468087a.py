"""Pentagon-free graph with independence number < 9 for R(C5,K9) lower bound."""
# EVOLVE-BLOCK-START

def construct():
    # Construction: 8 disjoint copies of K4.
    # Each K4 has 6 edges and is C5-free.
    # Independence number of each K4 is 1, so alpha(G) = 8 < 9.
    # Total vertices: 8 * 4 = 32, which certifies R(C5, K9) > 32.
    
    n = 32
    k = 4
    m = 8  # number of cliques

    edges = []
    for c in range(m):
        base = c * k
        for a in range(k):
            for b in range(a + 1, k):
                edges.append([base + a, base + b])

    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
