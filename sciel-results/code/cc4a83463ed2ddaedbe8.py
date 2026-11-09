"""5-colouring of K_n with no monochromatic C4 (R_5(C4)); maximise n."""
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before the 120 s wall-clock limit

    # ---- GF(4) arithmetic (2-bit integers, 2 = alpha, 3 = alpha+1) ----
    gf4_mult = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
        (1, 0): 0, (1, 1): 1, (1, 2): 2, (1, 3): 3,
        (2, 0): 0, (2, 1): 2, (2, 2): 3, (2, 3): 2,
        (3, 0): 0, (3, 1): 3, (3, 2): 2, (3, 3): 1,
    }
    gf4_inv = {1: 1, 2: 3, 3: 2}

    def gf4_add(a, b):
        return a ^ b

    def gf4_mul(a, b):
        return gf4_mult[(a, b)]

    def gf4_div(a, b):
        return gf4_mul(a, gf4_inv[b])

    # ---- GF(4)^2 construction: 16 vertices, 5 colours ----
    n = 16
    colors = []
    for i in range(n):
        ai, bi = i // 4, i % 4
        for j in range(i + 1, n):
            aj, bj = j // 4, j % 4
            d1 = gf4_add(ai, aj)
            d2 = gf4_add(bi, bj)
            if d1 == 0:
                c = 0          # vertical direction
            elif d2 == 0:
                c = 1          # horizontal direction
            else:
                slope = gf4_div(d2, d1)
                c = 2 + (slope - 1)   # slopes 1,2,3 -> colours 2,3,4
            colors.append(c)

    # ---- Fast C4 verification via bitmask common-neighbour counts ----
    def c4_free(n, colors):
        # neighbour bitmask per vertex, per colour
        nb = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                nb[c][i] |= 1 << j
                nb[c][j] |= 1 << i
                idx += 1
        for c in range(5):
            nbc = nb[c]
            for i in range(n):
                mi = nbc[i]
                for j in range(i + 1, n):
                    # a C4 exists iff two vertices share >=2 common neighbours
                    if (mi & nbc[j]).bit_count() >= 2:
                        return False
        return True

    if c4_free(n, colors):
        return {'n': n, 'colors': colors}

    # ---- Fallback: known-valid seed (K11 split into 5 Hamiltonian cycles) ----
    if time.time() < deadline:
        n = 11
        return {'n': n,
                'colors': [min(j - i, n - j + i) - 1
                           for i in range(n) for j in range(i + 1, n)]}

    # absolute last resort (should be unreachable)
    return {'n': 1, 'colors': [0]}
# EVOLVE-BLOCK-END
