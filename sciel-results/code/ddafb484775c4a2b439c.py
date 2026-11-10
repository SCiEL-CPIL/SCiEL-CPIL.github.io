"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill

    n = 16
    edges = []
    for u in range(n // 2):
        for v in range(n // 2, n):
            edges.append([u, v])

    # K(8,8) is optimal: 64 edges, triangle-free (Turán bound),
    # diameter 2, and every edge critical (adjacent vertices share
    # no common neighbor, so removing any edge makes that pair dist 3).
    return edges

# EVOLVE-BLOCK-END
