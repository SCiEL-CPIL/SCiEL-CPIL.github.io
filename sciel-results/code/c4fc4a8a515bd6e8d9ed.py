import time
import random

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = 2, 4, 6
    # rank = n * m * p = 48 for schoolbook
    # But we can use block-strassen: split into 3 blocks of 2x2x2, each rank 7 => 21
    # Wait, 2x4 * 4x6: the 4x4 middle is split into 2x2 blocks.
    # A is 2x4 = [A1 A2] where A1, A2 are 2x2
    # B is 4x6 = [B1; B2] where B1, B2 are 2x3
    # C = A1*B1 + A2*B2 (2x6)
    # We can use Strassen on 2x2 * 2x3? No, Strassen is for 2x2 * 2x2.
    # Let's split B into 3 blocks of 2x2: B = [B11 B12 B13]
    # C = A1*B11 + A2*B21 + A1*B12 + A2*B22 + A1*B13 + A2*B23
    # That's 6 blocks of 2x2 * 2x2. Each rank 7 => 42.
    # Can we do better?
    # 2x4 * 4x6 tensor rank. The schoolbook is 48.
    # Block strassen gives 42.
    # Let's verify 42 is correct and try to see if we can reduce further.
    # Actually, we can use the fact that A is 2x4 and B is 4x6.
    # The tensor is M_{2,4,6}.
    # Rank of M_{2,4,6} <= 42 via block strassen.
    # Is there a better bound?
    # For M_{m,n,p}, rank <= m*n*p.
    # Strassen for M_{2,2,2} is 7.
    # For M_{2,4,6}, we can partition into 6 sub-tensors M_{2,2,2}, each rank 7, total 42.
    # Let's stick with 42 and make sure the construction is correct.

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Each strassen factor (u, v, w) where u, v, w are linear combos of A, B, C elements.
    # We represent each as a list of (index, coeff) where index is 0..3 for 2x2 matrices.
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11

    strassen_factors = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),           # M2
        ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),           # M3
        ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),           # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),           # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),           # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),           # M7
    ]

    # 6 blocks: A_i * B_j for i in {0,1}, j in {0,1,2}
    # A block i: rows 0-1, cols 2*i..2*i+1
    # B block j: rows 2*j..2*j+1, cols 0-1
    # C block: rows 0-1, cols 2*j..2*j+1

    col = 0
    for i in range(2):
        for j in range(3):
            # A local index mapping: local (r,c) -> global (r, 2*i+c)
            # A global index: r * m + (2*i + c)
            # B local index mapping: local (r,c) -> global (2*j+r, c)
            # B global index: (2*j + r) * p + c
            # C local index mapping: local (r,c) -> global (r, 2*j+c)
            # C global index: (2*j + c) * n + r

            a_map = {0: 0*m + (2*i+0), 1: 0*m + (2*i+1), 2: 1*m + (2*i+0), 3: 1*m + (2*i+1)}
            b_map = {0: (2*j+0)*p + 0, 1: (2*j+0)*p + 1, 2: (2*j+1)*p + 0, 3: (2*j+1)*p + 1}
            w_map = {0: (2*j+0)*n + 0, 1: (2*j+0)*n + 1, 2: (2*j+1)*n + 0, 3: (2*j+1)*n + 1}

            for u_coeffs, v_coeffs, w_coeffs in strassen_factors:
                for idx, coeff in u_coeffs:
                    U[a_map[idx]][col] += coeff
                for idx, coeff in v_coeffs:
                    V[b_map[idx]][col] += coeff
                for idx, coeff in w_coeffs:
                    W[w_map[idx]][col] += coeff
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
