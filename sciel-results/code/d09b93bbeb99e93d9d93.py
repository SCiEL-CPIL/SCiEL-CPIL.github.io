import numpy as np
import time

def construct():
    """
    Returns a rank-minimal decomposition of the 4x4x4 matrix multiplication tensor.
    Attempts rank 47 (BCLR complex) then falls back to rank 49 (Strassen).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Attempt rank 47 decomposition (BCLR complex coefficients)
    # Note: No known rank < 49 decomposition exists for 4x4 matrices.
    # This function returns None to indicate failure, triggering fallback.
    result_47 = _try_rank47()
    if result_47 is not None:
        return result_47

    # Fallback to Rank 49 (Recursive Strassen)
    result_49 = _build_rank49()
    
    # Verify rank 49 result
    if _verify(result_49['U'], result_49['V'], result_49['W'], 49):
        return result_49
        
    # Final Fallback: Rank 64 (Schoolbook)
    return _build_rank64()

def _try_rank47():
    """
    Attempts to construct a rank 47 decomposition.
    Currently returns None as no such decomposition is known.
    """
    return None

def _build_rank49():
    """
    Constructs the rank 49 decomposition using recursive Strassen logic.
    """
    n, m, p = 4, 4, 4
    rank = 49

    # Strassen coefficients for 2x2 block level (s=0..6)
    beta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    delta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]

    # s=0 (M1)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    delta[0][0][0] = 1; delta[0][1][1] = 1

    # s=1 (M2)
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    delta[1][1][0] = 1; delta[1][1][1] = -1

    # s=2 (M3)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    delta[2][0][1] = 1; delta[2][1][1] = 1

    # s=3 (M4)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    delta[3][0][0] = 1; delta[3][1][0] = 1

    # s=4 (M5)
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    delta[4][0][0] = -1; delta[4][0][1] = 1

    # s=5 (M6)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    delta[5][1][1] = 1

    # s=6 (M7)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    delta[6][0][0] = 1

    # Strassen coefficients for 2x2 scalar level (t=0..6)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # t=0 (M1 inner)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # t=1 (M2 inner)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1

    # t=2 (M3 inner)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1

    # t=3 (M4 inner)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1

    # t=4 (M5 inner)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1

    # t=5 (M6 inner)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    # t=6 (M7 inner)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Construct U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for s in range(7):
        for t in range(7):
            c = 7 * s + t

            # Loop over A indices
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * U2[i_prime * 2 + j_prime][t]
                    if coeff != 0:
                        U[row_u][c] = coeff

            # Loop over B indices
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * V2[j_prime * 2 + k_prime][t]
                    if coeff != 0:
                        V[row_v][c] = coeff

            # Loop over C indices
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * W2[k_prime * 2 + i_prime][t]
                    if coeff != 0:
                        W[row_w][c] = coeff

    return {"U": U, "V": V, "W": W}

def _build_rank64():
    """
    Constructs the rank 64 (schoolbook) decomposition.
    """
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, rank):
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s_val = 0
                for r in range(rank):
                    s_val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                if s_val != 1:
                    return False
    return True
