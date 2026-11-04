"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Laderman's algorithm for 3x3x3 matrix multiplication (rank 23)
    # The 23 terms are defined by their linear forms in A, B and contributions to C.
    # Each term r has:
    #   U: dict {(i,j): coeff} for A[i,j]
    #   V: dict {(j,k): coeff} for B[j,k]
    #   W: dict {(k,i): coeff} for C[k,i] (output index)
    
    def make_term(U_dict, V_dict, W_dict):
        return (U_dict, V_dict, W_dict)
    
    # Correct Laderman 3x3x3 decomposition - 23 terms
    # Using 0-indexed: i,j,k in {0,1,2}
    # The key insight: for each term, the W coefficients must be such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    
    terms_3x3 = []
    
    # t1: (a00+a22)(b00+b22) -> c00, c02, c20, c22
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}
    ))
    
    # t2: (a00-a22)(b00-b22) -> c00, c02, c20, c22
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (0,2): -1, (2,0): -1, (2,2): 1}
    ))
    
    # t3: (a02+a20)(b02+b20) -> c00, c02, c20, c22
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): 1},
        {(0,2): 1, (2,0): 1},
        {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}
    ))
    
    # t4: (a02-a20)(b02-b20) -> c00, c02, c20, c22
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): -1},
        {(0,2): 1, (2,0): -1},
        {(0,0): 1, (0,2): -1, (2,0): -1, (2,2): 1}
    ))
    
    # t5: (a00+a11+a22)(b00+b11+b22) -> c00, c01, c02, c10, c11, c12, c20, c21, c22
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    
    # t6: (a01+a10+a22)(b01+b10+b22) -> c00, c01, c02, c10, c11, c12, c20, c21, c22
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    
    # t7: (a02+a12+a22)(b02+b12+b22) -> c00, c01, c02, c10, c11, c12, c20, c21, c22
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    
    # t8: (a11+a22)(b11+b22) -> c11, c12, c21, c22
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1}
    ))
    
    # t9: (a11-a22)(b11-b22) -> c11, c12, c21, c22
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (1,2): -1, (2,1): -1, (2,2): 1}
    ))
    
    # t10: (a12+a21)(b12+b21) -> c11, c12, c21, c22
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): 1},
        {(1,2): 1, (2,1): 1},
        {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1}
    ))
    
    # t11: (a12-a21)(b12-b21) -> c11, c12, c21, c22
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): -1},
        {(1,2): 1, (2,1): -1},
        {(1,1): 1, (1,2): -1, (2,1): -1, (2,2): 1}
    ))
    
    # t12: (a00+a11)(b00+b11) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t13: (a00-a11)(b00-b11) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # t14: (a01+a10)(b01+b10) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1},
        {(0,1): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t15: (a01-a10)(b01-b10) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): -1},
        {(0,1): 1, (1,0): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # t16: (a02+a12)(b02+b12) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1},
        {(0,2): 1, (1,2): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t17: (a02-a12)(b02-b12) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): -1},
        {(0,2): 1, (1,2): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # t18: (a20+a10)(b20+b10) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): 1},
        {(2,0): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t19: (a20-a10)(b20-b10) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): -1},
        {(2,0): 1, (1,0): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # t20: (a21+a11)(b21+b11) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): 1},
        {(2,1): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t21: (a21-a11)(b21-b11) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): -1},
        {(2,1): 1, (1,1): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # t22: (a22+a02)(b22+b02) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): 1},
        {(2,2): 1, (0,2): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    
    # t23: (a22-a02)(b22-b02) -> c00, c01, c10, c11
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): -1},
        {(2,2): 1, (0,2): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    
    # Verify the 3x3x3 decomposition
    def verify_3x3(terms):
        T_target = np.zeros((9, 9, 9), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        U = np.zeros((9, len(terms)), dtype=np.float64)
        V = np.zeros((9, len(terms)), dtype=np.float64)
        W = np.zeros((9, len(terms)), dtype=np.float64)
        
        for r, (U_dict, V_dict, W_dict) in enumerate(terms):
            for (i, j), coeff in U_dict.items():
                U[i*3+j, r] = coeff
            for (j, k), coeff in V_dict.items():
                V[j*3+k, r] = coeff
            for (k, i), coeff in W_dict.items():
                W[k*3+i, r] = coeff
        
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_comp, T_target, atol=1e-10)
    
    # If verification fails, use schoolbook
    if not verify_3x3(terms_3x3):
        rank_schoolbook = n * m * p
        U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Use Laderman for both 3x3 blocks of the 3x3x6 problem
    rank = 46
    U_mat = np.zeros((n * m, rank), dtype=np.float64)
    V_mat = np.zeros((m * p, rank), dtype=np.float64)
    W_mat = np.zeros((p * n, rank), dtype=np.float64)
    
    # First block (k=0,1,2)
    for r in range(23):
        U_dict, V_dict, W_dict = terms_3x3[r]
        for (i, j), coeff in U_dict.items():
            U_mat[i * m + j, r] = coeff
        for (j, k), coeff in V_dict.items():
            V_mat[j * p + k, r] = coeff
        for (k, i), coeff in W_dict.items():
            W_mat[k * n + i, r] = coeff
    
    # Second block (k=3,4,5)
    for r in range(23):
        U_dict, V_dict, W_dict = terms_3x3[r]
        for (i, j), coeff in U_dict.items():
            U_mat[i * m + j, r + 23] = coeff
        for (j, k), coeff in V_dict.items():
            V_mat[j * p + (k + 3), r + 23] = coeff
        for (k, i), coeff in W_dict.items():
            W_mat[(k + 3) * n + i, r + 23] = coeff
    
    # Verify the full 3x3x6 decomposition
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    if not np.allclose(T_comp, T_target, atol=1e-10):
        rank_schoolbook = n * m * p
        U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Convert to lists of floats (integer/half-integer entries)
    U_list = [[float(x) for x in row] for row in U_mat]
    V_list = [[float(x) for x in row] for row in V_mat]
    W_list = [[float(x) for x in row] for row in W_mat]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
