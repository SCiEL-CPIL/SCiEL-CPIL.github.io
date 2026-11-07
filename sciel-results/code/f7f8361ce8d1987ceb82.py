"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    half = n // 2
    # Partition 1: 0 to half-1
    # Partition 2: half to n-1
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
