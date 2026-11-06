"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _validate(color_map):
    """Check every K4 spans at least 5 distinct colours."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            if x > y:
                x, y = y, x
            cols.add(color_map[(x, y)])
        if len(cols) < 5:
            return False
    return True

def construct():
    t0 = time.time()
    deadline = 110  # safety margin before 120 s kill

    # 1-factorization of K16: 15 colours.
    # Vertex 15 is the "centre"; vertices 0..14 sit on a cycle.
    # Factor i  =  {(15, i)}  ∪  {(i+j mod 15, i-j mod 15) : j = 1..7}
    color_map = {}
    for i in range(15):
        color_map[(15, i)] = i
        color_map[(i, 15)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            color_map[(a, b)] = i

    if _validate(color_map):
        result = []
        for i in range(16):
            for j in range(i + 1, 16):
                result.append(color_map[(i, j)])
        return result

    # Fallback: (i + j) mod 17  –  known valid, 17 colours.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
