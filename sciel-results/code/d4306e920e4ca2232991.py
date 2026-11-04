import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def build_3x3x3_block():
        """Returns list of (u, v, w) dicts for a 3x3x3 block."""
        terms = []

        # Strassen terms for the 2x2x2 sub-block (i,j,k in {0,1})
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]

        # Verify Strassen correctness for 2x2x2
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for u, v, w in strassen_terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            s += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break

        if valid:
            terms.extend(strassen_terms)
        else:
            # Fallback to schoolbook for 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        terms.append(( {(i,j):1}, {(j,k):1}, {(i,k):1} ))

        # Schoolbook for remaining 19 terms (at least one index is 2)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    terms.append(( {(i,j):1}, {(j,k):1}, {(i,k):1} ))

        return terms

    block1_terms = build_3x3x3_block()
    block2_terms = build_3x3x3_block()

    total_rank = len(block1_terms) + len(block2_terms)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # Block 1: k in 0,1,2
    for u, v, w in block1_terms:
        for (i, j), c in u.items():
            U[i * m + j][col] += c
        for (j, k), c in v.items():
            V[j * p + k][col] += c
        for (i, k), c in w.items():
            W[k * n + i][col] += c
        col += 1

    # Block 2: k in 3,4,5
    for u, v, w in block2_terms:
        for (i, j), c in u.items():
            U[i * m + j][col] += c
        for (j, k_prime), c in v.items():
            k = k_prime + 3
            V[j * p + k][col] += c
        for (i, k_prime), c in w.items():
            k = k_prime + 3
            W[k * n + i][col] += c
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
