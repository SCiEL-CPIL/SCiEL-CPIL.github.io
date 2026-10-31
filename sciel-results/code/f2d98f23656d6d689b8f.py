import numpy as np

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    n, m, p = 4, 4, 4
    rank = 49  # 7 * 7: outer Strassen (7 terms) x inner Strassen (7 terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 coefficients (7 terms each for A, B, C linear forms)
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th outer/inner A-form
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th outer/inner B-form
    # c_coeffs[s][i][k]: coefficient of C[i][k] in s-th outer/inner C-form
    #
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5

    a_coeffs = [
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ]

    c_coeffs = [
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ]

    # Build U, V, W using the tensor product structure:
    # For 4x4, we have 2x2 blocks.
    # Global index (i, j) where i, j in 0..3.
    # Block index (I, J) where I, J in 0..1.
    # Local index (i', j') where i', j' in 0..1.
    # i = 2*I + i', j = 2*J + j'
    # 
    # U[i*4+j, col] = a_coeffs[s][I][J] * a_coeffs[t][i'][j']
    # V[j*4+k, col] = b_coeffs[s][J][K] * b_coeffs[t][j'][k']
    # W[k*4+i, col] = c_coeffs[s][I][K] * c_coeffs[t][i'][k']
    # 
    # Note: The W index in the problem description is W[k*n+i, r].
    # Here n=4. So W index = k*4 + i.
    # In our block decomposition:
    # Global row of C is i, global col of C is k.
    # Block row I, block col K.
    # Local row i', local col k'.
    # i = 2*I + i', k = 2*K + k'.
    # W index = k*4 + i = (2*K + k')*4 + (2*I + i') = 8*K + 4*k' + 2*I + i'.
    # 
    # The coefficient for W at this index for term (s, t) is:
    # c_coeffs[s][I][K] * c_coeffs[t][i'][k']
    # 
    # Let's iterate over s, t (0..6) and fill the matrices.

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    a_outer = a_coeffs[s][I][J]
                    if a_outer == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            a_inner = a_coeffs[t][ip][jp]
                            if a_inner == 0:
                                continue
                            coeff = a_outer * a_inner
                            if coeff != 0:
                                i = 2 * I + ip
                                j = 2 * J + jp
                                U[i * 4 + j][col] = coeff

            # Fill V
            for J in range(2):
                for K in range(2):
                    b_outer = b_coeffs[s][J][K]
                    if b_outer == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            b_inner = b_coeffs[t][jp][kp]
                            if b_inner == 0:
                                continue
                            coeff = b_outer * b_inner
                            if coeff != 0:
                                j = 2 * J + jp
                                k = 2 * K + kp
                                V[j * 4 + k][col] = coeff

            # Fill W
            for I in range(2):
                for K in range(2):
                    c_outer = c_coeffs[s][I][K]
                    if c_outer == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            c_inner = c_coeffs[t][ip][kp]
                            if c_inner == 0:
                                continue
                            coeff = c_outer * c_inner
                            if coeff != 0:
                                i = 2 * I + ip
                                k = 2 * K + kp
                                # W index = k * 4 + i
                                W[k * 4 + i][col] = coeff

    return {"U": U, "V": V, "W": W}
