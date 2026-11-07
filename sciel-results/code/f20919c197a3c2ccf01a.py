import time

# EVOLVE-BLOCK-START

def construct():
    n = 16
    half = n // 2
    edges = []
    # Partition vertices into two sets of size 8
    # Set U: 0..7, Set V: 8..15
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
