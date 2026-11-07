"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    h = n // 2
    out = []
    for a in range(h):
        for b in range(h, n):
            out.append([a, b])
    return out

# EVOLVE-BLOCK-END
