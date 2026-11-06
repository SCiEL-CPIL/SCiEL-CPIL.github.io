"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Base K32 coloring
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Return the base K32 coloring
    return {'n': 32, 'colors': base_colors}
# EVOLVE-BLOCK-END
