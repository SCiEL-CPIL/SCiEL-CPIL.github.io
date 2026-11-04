"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    deadline = start_time + 145  # 5s safety margin
    
    # Laderman's algorithm for 3x3x3 matrix multiplication (rank 23)
    # C = A * B where A, B, C are 3x3
    # The 23 multiplications are defined by the following 23 terms:
    # Each term is a product of a linear form in A, a linear form in B, and contributes to C
    
    # Laderman's 3x3 matrix multiplication:
    # m1 = (a11+a33)(b11+b33)
    # m2 = (a11-a33)(b11-b33)  
    # m3 = (a13+a31)(b13+b31)
    # m4 = (a13-a31)(b13-b31)
    # m5 = (a11+a22+a33)(b11+b22+b33)
    # m6 = (a12+a21+a33)(b12+b21+b33)
    # m7 = (a13+a23+a33)(b13+b23+b33)
    # m8 = (a22+a33)(b22+b33)
    # m9 = (a22-a33)(b22-b33)
    # m10 = (a23+a32)(b23+b32)
    # m11 = (a23-a32)(b23-b32)
    # m12 = (a11+a22)(b11+b22)
    # m13 = (a11-a22)(b11-b22)
    # m14 = (a12+a21)(b12+b21)
    # m15 = (a12-a21)(b12-b21)
    # m16 = (a13+a23)(b13+b23)
    # m17 = (a13-a23)(b13-b23)
    # m18 = (a31+a21)(b31+b21)
    # m19 = (a31-a21)(b31-b21)
    # m20 = (a32+a22)(b32+b22)
    # m21 = (a32-a22)(b32-b22)
    # m22 = (a33+a13)(b33+b13)
    # m23 = (a33-a13)(b33-b13)
    
    # Actually, let me use a more standard formulation. 
    # Laderman's algorithm can be expressed as 23 rank-1 terms.
    # Let me define them explicitly.
    
    # For 3x3x3, the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # We need to find U (9x23), V (9x23), W (9x23) such that
    # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all valid (i,j,k)
    
    # Let me define Laderman's 23 terms explicitly.
    # Each term r has:
    #   U contributions: for each (i,j), U[i*3+j, r]
    #   V contributions: for each (j,k), V[j*3+k, r]
    #   W contributions: for each (k,i), W[k*3+i, r]
    
    # I'll use a known explicit formulation of Laderman's algorithm.
    # The 23 multiplications and their contributions:
    
    laderman_terms = []
    
    # Term 1: (a11+a33)(b11+b33) -> contributes to c11, c13, c31, c33
    # U: (0,0)=1, (2,2)=1
    # V: (0,0)=1, (2,2)=1
    # W: (0,0)=1, (2,2)=1, (0,2)=-1, (2,0)=-1  (need to check signs)
    # Actually, let me be more careful.
    
    # Let me use a different approach: I'll define the 23 terms by their 
    # linear forms in A, B and their contributions to C.
    
    # Standard Laderman 3x3x3 decomposition:
    # The 23 terms are (using 0-indexed i,j,k for 3x3 matrices):
    
    # I'll define each term as (U_coeffs, V_coeffs, W_coeffs)
    # where U_coeffs is a dict {(i,j): coeff}, etc.
    
    def make_term(U_dict, V_dict, W_dict):
        """Create a term from dictionaries of coefficients."""
        return (U_dict, V_dict, W_dict)
    
    # Laderman's 3x3x3 - 23 terms
    # Using 0-indexed: i,j,k in {0,1,2}
    
    terms_3x3 = []
    
    # t1 = (a00+a22)(b00+b22)
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (2,2): 1},
        {(0,0): 1, (2,2): 1}
    ))
    
    # t2 = (a00-a22)(b00-b22)
    terms_3x3.append(make_term(
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (2,2): -1},
        {(0,0): 1, (2,2): -1}
    ))
    
    # t3 = (a02+a20)(b02+b20)
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): 1},
        {(0,2): 1, (2,0): 1},
        {(0,2): 1, (2,0): 1}
    ))
    
    # t4 = (a02-a20)(b02-b20)
    terms_3x3.append(make_term(
        {(0,2): 1, (2,0): -1},
        {(0,2): 1, (2,0): -1},
        {(0,2): 1, (2,0): -1}
    ))
    
    # t5 = (a00+a11+a22)(b00+b11+b22)
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (1,1): 1, (2,2): 1},
        {(0,0): 1, (1,1): 1, (2,2): 1}
    ))
    
    # t6 = (a01+a10+a22)(b01+b10+b22)
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,1): 1, (1,0): 1, (2,2): 1},
        {(0,1): 1, (1,0): 1, (2,2): 1}
    ))
    
    # t7 = (a02+a12+a22)(b02+b12+b22)
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1}
    ))
    
    # t8 = (a11+a22)(b11+b22)
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1}
    ))
    
    # t9 = (a11-a22)(b11-b22)
    terms_3x3.append(make_term(
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (2,2): -1},
        {(1,1): 1, (2,2): -1}
    ))
    
    # t10 = (a12+a21)(b12+b21)
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): 1},
        {(1,2): 1, (2,1): 1},
        {(1,2): 1, (2,1): 1}
    ))
    
    # t11 = (a12-a21)(b12-b21)
    terms_3x3.append(make_term(
        {(1,2): 1, (2,1): -1},
        {(1,2): 1, (2,1): -1},
        {(1,2): 1, (2,1): -1}
    ))
    
    # t12 = (a00+a11)(b00+b11)
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # t13 = (a00-a11)(b00-b11)
    terms_3x3.append(make_term(
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (1,1): -1},
        {(0,0): 1, (1,1): -1}
    ))
    
    # t14 = (a01+a10)(b01+b10)
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): 1},
        {(0,1): 1, (1,0): 1},
        {(0,1): 1, (1,0): 1}
    ))
    
    # t15 = (a01-a10)(b01-b10)
    terms_3x3.append(make_term(
        {(0,1): 1, (1,0): -1},
        {(0,1): 1, (1,0): -1},
        {(0,1): 1, (1,0): -1}
    ))
    
    # t16 = (a02+a12)(b02+b12)
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): 1},
        {(0,2): 1, (1,2): 1},
        {(0,2): 1, (1,2): 1}
    ))
    
    # t17 = (a02-a12)(b02-b12)
    terms_3x3.append(make_term(
        {(0,2): 1, (1,2): -1},
        {(0,2): 1, (1,2): -1},
        {(0,2): 1, (1,2): -1}
    ))
    
    # t18 = (a20+a10)(b20+b10)
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): 1},
        {(2,0): 1, (1,0): 1},
        {(2,0): 1, (1,0): 1}
    ))
    
    # t19 = (a20-a10)(b20-b10)
    terms_3x3.append(make_term(
        {(2,0): 1, (1,0): -1},
        {(2,0): 1, (1,0): -1},
        {(2,0): 1, (1,0): -1}
    ))
    
    # t20 = (a21+a11)(b21+b11)
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): 1},
        {(2,1): 1, (1,1): 1},
        {(2,1): 1, (1,1): 1}
    ))
    
    # t21 = (a21-a11)(b21-b11)
    terms_3x3.append(make_term(
        {(2,1): 1, (1,1): -1},
        {(2,1): 1, (1,1): -1},
        {(2,1): 1, (1,1): -1}
    ))
    
    # t22 = (a22+a02)(b22+b02)
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): 1},
        {(2,2): 1, (0,2): 1},
        {(2,2): 1, (0,2): 1}
    ))
    
    # t23 = (a22-a02)(b22-b02)
    terms_3x3.append(make_term(
        {(2,2): 1, (0,2): -1},
        {(2,2): 1, (0,2): -1},
        {(2,2): 1, (0,2): -1}
    ))
    
    # Now I need to verify this is actually correct for 3x3x3.
    # Let me construct the full tensor and check.
    
    def verify_3x3(terms):
        """Verify that the terms correctly decompose the 3x3x3 multiplication tensor."""
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
    
    # Check if our Laderman terms are correct
    if not verify_3x3(terms_3x3):
        # If the simple formulation doesn't work, fall back to schoolbook
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
        
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # The 3x3x3 decomposition works. Now use it for 3x3x6.
    # Split B (3x6) into B1 (3x3, columns 0-2) and B2 (3x3, columns 3-5)
    # C = A*B1 + A*B2
    # Use Laderman for each, giving rank 23+23 = 46
    
    rank = 46
    U_mat = np.zeros((n * m, rank), dtype=np.float64)
    V_mat = np.zeros((m * p, rank), dtype=np.float64)
    W_mat = np.zeros((p * n, rank), dtype=np.float64)
    
    # For the first block (k=0,1,2):
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*6 + k (k in 0,1,2)
    # W index: k*n + i = k*3 + i (k in 0,1,2)
    
    for r in range(23):
        U_dict, V_dict, W_dict = terms_3x3[r]
        for (i, j), coeff in U_dict.items():
            U_mat[i * m + j, r] = coeff
        for (j, k), coeff in V_dict.items():
            V_mat[j * p + k, r] = coeff  # k is 0,1,2
        for (k, i), coeff in W_dict.items():
            W_mat[k * n + i, r] = coeff  # k is 0,1,2
    
    # For the second block (k=3,4,5):
    # U index: i*m + j = i*3 + j (same)
    # V index: j*p + k = j*6 + k (k in 3,4,5)
    # W index: k*n + i = k*3 + i (k in 3,4,5)
    
    for r in range(23):
        U_dict, V_dict, W_dict = terms_3x3[r]
        for (i, j), coeff in U_dict.items():
            U_mat[i * m + j, r + 23] = coeff
        for (j, k), coeff in V_dict.items():
            V_mat[j * p + (k + 3), r + 23] = coeff  # k+3 is 3,4,5
        for (k, i), coeff in W_dict.items():
            W_mat[(k + 3) * n + i, r + 23] = coeff  # k+3 is 3,4,5
    
    # Verify the full decomposition
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
        # Fallback to schoolbook
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
        
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_mat]
    V_list = [[float(x) for x in row] for row in V_mat]
    W_list = [[float(x) for x in row] for row in W_mat]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
