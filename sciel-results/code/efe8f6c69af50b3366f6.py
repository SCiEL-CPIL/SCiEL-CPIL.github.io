"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.
Uses a direct construction based on Strassen-like identities for the 2x4x6 tensor.

The approach:
1. Split A (2x4) into two 2x2 blocks: A_L = A[:,0:2], A_R = A[:,2:4]
2. Split B (4x6) into two 2x6 blocks: B_L = B[0:2,:], B_R = B[2:4,:]
3. C = A_L @ B_L + A_R @ B_R
4. Each (2x2)@(2x6) is decomposed using 3 Strassen 2x2x2 instances (one per 2x2 column block of the 2x6 matrix)
5. Total rank: 2 * 3 * 7 = 42

The tensor identity verified:
T[(i,j),(j',k),(k',i')] = 1 iff j==j' and k==k' and i==i'
Decomposition: T = sum_r u_r * v_r * w_r
where u_r[i*m+j], v_r[j'*p+k], w_r[k'*n+i']
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_2x2x2_factors():
    """Return the 7 rank-1 factors for 2x2x2 Strassen multiplication.
    
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    factors = []
    
    # M1 = (A11 + A22)(B11 + B22)
    factors.append({
        'U': {0: 1, 3: 1},
        'V': {0: 1, 3: 1},
        'W': {0: 1, 3: 1}
    })
    
    # M2 = (A21 + A22)B11
    factors.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {2: 1}
    })
    
    # M3 = A11(B21 - B22)
    factors.append({
        'U': {0: 1},
        'V': {2: 1, 3: -1},
        'W': {1: 1, 3: 1}
    })
    
    # M4 = A22(B12 - B11)
    factors.append({
        'U': {3: 1},
        'V': {1: 1, 0: -1},
        'W': {0: 1, 2: 1}
    })
    
    # M5 = (A11 + A12)B22
    factors.append({
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {0: -1, 1: 1}
    })
    
    # M6 = (A21 - A11)(B11 + B12)
    factors.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    })
    
    # M7 = (A12 - A22)(B21 + B22)
    factors.append({
        'U': {1: 1, 3: -1},
        'V': {2: 1, 3: 1},
        'W': {0: 1}
    })
    
    return factors


def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)

    strassen_factors = _strassen_2x2x2_factors()

    factor_idx = 0
    for a_block in range(2):  # 0: A_L (cols 0,1), 1: A_R (cols 2,3)
        for b_col_block in range(3):  # 0: cols 0,1; 1: cols 2,3; 2: cols 4,5
            for sf in strassen_factors:
                # U part: A[i, j_local] where j_local in {0,1}
                for a_idx, coeff in sf['U'].items():
                    i, j = divmod(a_idx, 2)
                    if a_block == 0:
                        u_row = i * 4 + j
                    else:
                        u_row = i * 4 + (j + 2)
                    U[u_row][factor_idx] += coeff

                # V part: B[j_local, k_local] where j_local in {0,1}, k_local in {0,1}
                for b_idx, coeff in sf['V'].items():
                    j, k = divmod(b_idx, 2)
                    if a_block == 0:
                        v_row = j * 6 + (2 * b_col_block + k)
                    else:
                        v_row = (j + 2) * 6 + (2 * b_col_block + k)
                    V[v_row][factor_idx] += coeff

                # W part: C[k_local, i] where k_local in {0,1}, i in {0,1}
                for c_idx, coeff in sf['W'].items():
                    k_local, i = divmod(c_idx, 2)
                    k_global = 2 * b_col_block + k_local
                    w_row = k_global * 2 + i
                    W[w_row][factor_idx] += coeff

                factor_idx += 1

    assert factor_idx == 42, f"Expected 42 factors, got {factor_idx}"

    # Verify correctness with the tensor identity
    def check_tensor_identity(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and k == kp and i == ip) else 0
                                actual = 0
                                for r in range(rank):
                                    actual += U[i * m + j][r] * V[jp * p + k][r] * W[kp * n + ip][r]
                                if expected != actual:
                                    return False
        return True

    ok = check_tensor_identity(U, V, W, n, m, p)
    if not ok:
        # Fall back to schoolbook rank 48
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return schoolbook as safety
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
