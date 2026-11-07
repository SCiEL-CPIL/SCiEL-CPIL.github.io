"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    a = n // 2
    out = []
    for u in range(a):
        for v in range(a, n):
            out.append([u, v])
    return out
# EVOLVE-BLOCK-END
