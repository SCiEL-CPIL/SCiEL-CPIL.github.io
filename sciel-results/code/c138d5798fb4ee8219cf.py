"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def make_vec(indices, coeff=1):
        d = {}
        for idx in indices:
            d[idx] = d.get(idx, 0) + coeff
        return d
    
    def add_vecs(*vecs):
        result = {}
        for v in vecs:
            for k, val in v.items():
                result[k] = result.get(k, 0) + val
        return result
    
    def neg_vec(v):
        return {k: -val for k, val in v.items()}
    
    # Block index sets for 4x4 A, 4x6 B, 4x6 C (but C is 6x4 in W)
    # A[i,j] -> U index i*4+j
    # B[j,k] -> V index j*6+k
    # C[k,i] -> W index k*4+i (trace convention)
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 6, 7]      # B[0:2, 0:2]
    B12 = [2, 3, 8, 9]      # B[0:2, 2:4]
    B21 = [12, 13, 18, 19]  # B[2:4, 0:2]
    B22 = [14, 15, 20, 21]  # B[2:4, 2:4]
    
    C11 = [0, 1, 4, 5]      # C[0:2, 0:2]
    C12 = [2, 3, 6, 7]      # C[0:2, 2:4]
    C21 = [8, 9, 12, 13]    # C[2:4, 0:2]
    C22 = [10, 11, 14, 15]  # C[2:4, 2:4]
    
    a11, a12, a21, a22 = [make_vec(x) for x in [A11, A12, A21, A22]]
    b11, b12, b21, b22 = [make_vec(x) for x in [B11, B12, B21, B22]]
    c11, c12, c21, c22 = [make_vec(x) for x in [C11, C12, C21, C22]]
    
    # Strassen 7 multiplications for 4x4 x 4x4 -> 4x4
    # M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
    add_term(add_vecs(a11, a22), add_vecs(b11, b22), add_vecs(c11, c22))
    # M2 = (A21+A22)B11 -> C21:+1, C22:-1
    add_term(add_vecs(a21, a22), b11, add_vecs(c21, neg_vec(c22)))
    # M3 = A11(B21-B22) -> C12:+1, C21:+1
    add_term(a11, add_vecs(b21, neg_vec(b22)), add_vecs(c12, c21))
    # M4 = A22(B12-B11) -> C11:+1, C21:+1
    add_term(a22, add_vecs(b12, neg_vec(b11)), add_vecs(c11, c21))
    # M5 = (A11+A12)B22 -> C11:-1, C12:+1, C22:+1
    add_term(add_vecs(a11, a12), b22, add_vecs(neg_vec(c11), c12, c22))
    # M6 = (A21-A11)(B11+B12) -> C21:+1, C22:+1
    add_term(add_vecs(a21, neg_vec(a11)), add_vecs(b11, b12), add_vecs(c21, c22))
    # M7 = (A12-A22)(B21+B22) -> C11:+1, C12:+1, C21:+1, C22:+1
    add_term(add_vecs(a12, neg_vec(a22)), add_vecs(b21, b22), add_vecs(c11, c12, c21, c22))
    
    # Part 2: C_right = A * B_right where B_right is 4x2 (cols 4-5)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_idx = i * 4 + j
                v_idx = j * 6 + (4 + k)
                w_idx = (4 + k) * 4 + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for idx, coeff in U_cols[col].items():
            U[idx][col] = coeff
        for idx, coeff in V_cols[col].items():
            V[idx][col] = coeff
        for idx, coeff in W_cols[col].items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
