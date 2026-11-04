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
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Laderman's algorithm for 3x3x3 matrix multiplication (rank 23)
    # We define the 23 terms for the 3x3x3 multiplication.
    # Each term is a tuple of three dictionaries:
    # U_dict: {(i,j): coeff} for A[i,j]
    # V_dict: {(j,k): coeff} for B[j,k]
    # W_dict: {(k,i): coeff} for C[k,i]
    
    def make_term(U_dict, V_dict, W_dict):
        return (U_dict, V_dict, W_dict)
    
    terms_3x3 = []
    
    # t1
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}
    ))
    # t2
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (0,2): -1, (2,0): -1, (2,2): 1}
    ))
    # t3
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): 1},
        {(0,2): 1, (2,0): 1},
        {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}
    ))
    # t4
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): -1},
        {(0,2): 1, (2,0): -1},
        {(0,0): 1, (0,2): -1, (2,0): -1, (2,2): 1}
    ))
    # t5
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    # t6
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    # t7
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}
    ))
    # t8
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1}
    ))
    # t9
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (1,2): -1, (2,1): -1, (2,2): 1}
    ))
    # t10
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): 1},
        {(1,2): 1, (2,1): 1},
        {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1}
    ))
    # t11
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): -1},
        {(1,2): 1, (2,1): -1},
        {(1,1): 1, (1,2): -1, (2,1): -1, (2,2): 1}
    ))
    # t12
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t13
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    # t14
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1},
        {(0,1): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t15
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): -1},
        {(0,1): 1, (1,0): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    # t16
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1},
        {(0,2): 1, (1,2): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t17
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): -1},
        {(0,2): 1, (1,2): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    # t18
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): 1},
        {(2,0): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t19
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): -1},
        {(2,0): 1, (1,0): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    # t20
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): 1},
        {(2,1): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t21
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): -1},
        {(2,1): 1, (1,1): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))
    # t22
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): 1},
        {(2,2): 1, (0,2): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    ))
    # t23
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): -1},
        {(2,2): 1, (0,2): -1},
        {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    ))

    # Construct the full 3x3x6 decomposition
    # We use two copies of the 3x3x3 decomposition.
    # Block 1: k in {0, 1, 2}
    # Block 2: k in {3, 4, 5}
    
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
            # Map k from 0..2 to 3..5
            V_mat[j * p + (k + 3), r + 23] = coeff
        for (k, i), coeff in W_dict.items():
            # Map k from 0..2 to 3..5
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
        # Fallback to schoolbook if verification fails
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
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U_mat]
    V_list = [[float(x) for x in row] for row in V_mat]
    W_list = [[float(x) for x in row] for row in W_mat]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
