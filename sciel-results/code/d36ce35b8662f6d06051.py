"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    if n != 6 or m != 6 or p != 6:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Decompose 6x6x6 as (2x2x2) kron (3x3x3)
    # 2x2x2 uses Strassen (rank 7)
    # 3x3x3 uses schoolbook (rank 27)
    # Total rank = 7 * 27 = 189
    # U shape: (36, 189), V shape: (36, 189), W shape: (36, 189)
    # U index: (i2*3+i3, j2*3+j3) where i2 in {0,1}, j2 in {0,1}, i3 in {0,1,2}, j3 in {0,1,2}
    # V index: (j2*3+j3, k2*3+k3) where j2 in {0,1}, k2 in {0,1}, j3 in {0,1,2}, k3 in {0,1,2}
    # W index: (k2*3+k3, i2*3+i3) where k2 in {0,1}, i2 in {0,1}, k3 in {0,1,2}, i3 in {0,1,2}

    # Strassen for 2x2:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B10-B11)
    # M3 = A11*(B01-B10)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # Strassen factors for 2x2:
    # For each Strassen step s in 0..6:
    # U2[a, b, s]: nonzero entries
    # V2[b, c, s]: nonzero entries
    # W2[c, a, s]: nonzero entries

    # Let's define the Strassen factors as lists of (index, value) for each step
    # U2 index: a*2 + b
    # V2 index: b*2 + c
    # W2 index: c*2 + a

    U2_steps = []
    V2_steps = []
    W2_steps = []

    # s=0: M0 = (A00+A11)*(B00+B11)
    # U: A00, A11
    U2_steps.append([(0*2+0, 1), (1*2+1, 1)])
    V2_steps.append([(0*2+0, 1), (1*2+1, 1)])
    W2_steps.append([(0*2+0, 1)])  # C00 += M0, C11 += M0

    # s=1: M1 = (A10+A11)*B00
    # U: A10, A11
    U2_steps.append([(1*2+0, 1), (1*2+1, 1)])
    V2_steps.append([(0*2+0, 1)])
    W2_steps.append([(1*2+0, 1)])  # C10 += M1

    # s=2: M2 = A00*(B10-B11)
    # U: A00
    U2_steps.append([(0*2+0, 1)])
    V2_steps.append([(1*2+0, 1), (1*2+1, -1)])
    W2_steps.append([(0*2+1, 1), (1*2+1, 1)])  # C01 += M2, C11 += M2

    # s=3: M3 = A11*(B01-B10)
    # U: A11
    U2_steps.append([(1*2+1, 1)])
    V2_steps.append([(0*2+1, 1), (1*2+0, -1)])
    W2_steps.append([(0*2+0, 1), (1*2+0, 1)])  # C00 += M3, C10 += M3

    # s=4: M4 = (A00+A01)*B11
    # U: A00, A01
    U2_steps.append([(0*2+0, 1), (0*2+1, 1)])
    V2_steps.append([(1*2+1, 1)])
    W2_steps.append([(0*2+1, -1), (1*2+1, -1)])  # C01 -= M4, C11 -= M4

    # s=5: M5 = (A10-A00)*(B00+B01)
    # U: A10, -A00
    U2_steps.append([(1*2+0, 1), (0*2+0, -1)])
    V2_steps.append([(0*2+0, 1), (0*2+1, 1)])
    W2_steps.append([(1*2+1, 1)])  # C11 += M5

    # s=6: M6 = (A01-A11)*(B10+B11)
    # U: A01, -A11
    U2_steps.append([(0*2+1, 1), (1*2+1, -1)])
    V2_steps.append([(1*2+0, 1), (1*2+1, 1)])
    W2_steps.append([(0*2+0, 1)])  # C00 += M6

    # Schoolbook for 3x3: 27 steps, each (i3, j3, k3)
    U3_steps = []
    V3_steps = []
    W3_steps = []
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                U3_steps.append([(i3*3+j3, 1)])
                V3_steps.append([(j3*3+k3, 1)])
                W3_steps.append([(k3*3+i3, 1)])

    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    for s2 in range(7):
        for s3 in range(27):
            r = s2 * 27 + s3

            # U: combine U2[s2] and U3[s3]
            for (u2_idx, u2_val) in U2_steps[s2]:
                a2 = u2_idx // 2
                b2 = u2_idx % 2
                for (u3_idx, u3_val) in U3_steps[s3]:
                    i3 = u3_idx // 3
                    j3 = u3_idx % 3
                    u_idx = (a2 * 3 + i3) * 3 + (b2 * 3 + j3)
                    U[u_idx][r] += u2_val * u3_val

            # V: combine V2[s2] and V3[s3]
            for (v2_idx, v2_val) in V2_steps[s2]:
                b2 = v2_idx // 2
                c2 = v2_idx % 2
                for (v3_idx, v3_val) in V3_steps[s3]:
                    j3 = v3_idx // 3
                    k3 = v3_idx % 3
                    v_idx = (b2 * 3 + j3) * 3 + (c2 * 3 + k3)
                    V[v_idx][r] += v2_val * v3_val

            # W: combine W2[s2] and W3[s3]
            for (w2_idx, w2_val) in W2_steps[s2]:
                c2 = w2_idx // 2
                a2 = w2_idx % 2
                for (w3_idx, w3_val) in W3_steps[s3]:
                    k3 = w3_idx // 3
                    i3 = w3_idx % 3
                    w_idx = (c2 * 3 + k3) * 3 + (a2 * 3 + i3)
                    W[w_idx][r] += w2_val * w3_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
