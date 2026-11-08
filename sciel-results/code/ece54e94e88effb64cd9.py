import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Strategy: Rank 23 for 3x3 core + Rank 9 for the 4th column = Rank 32
    # A is 3x3 (n x m), B is 3x4 (m x p)
    # C is 3x4 (n x p), C[i,k] = sum_j A[i,j]*B[j,k]
    # W convention: W[k*n + i, r]
    
    # 3x3 Strassen-like rank-23 algorithm for C[:3, :3] = A * B[:3, :3]
    # Then 9 schoolbook terms for C[:3, 3] = A * B[:3, 3]
    
    rank_23 = 23
    rank_9 = 9
    rank = rank_23 + rank_9  # 32
    
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 32
    
    # A indices: i in [0,1,2], j in [0,1,2] -> U[i*3+j, r]
    # B indices: j in [0,1,2], k in [0,1,2,3] -> V[j*4+k, r]
    # C indices: i in [0,1,2], k in [0,1,2,3] -> W[k*3+i, r]
    
    # ===== Part 1: Rank-23 algorithm for 3x3 * 3x3 -> 3x3 =====
    # Using the known 23-multiplication algorithm
    # C[i][k] for i,k in 0..2
    
    # Let me define the 23 bilinear forms explicitly.
    # A[i,j] accessed as U[i*3+j, c], B[j,k] as V[j*4+k, c]
    
    # The 23-term decomposition for 3x3 matrix mult:
    # I'll use the standard one from literature.
    
    # Define each of the 23 terms as (U_coeff, V_coeff, W_coeff)
    # Each term: U_vec (9 entries), V_vec (12 entries for 3x4, but only first 9 used for 3x3), W_vec (12 entries for 3x4, but only first 9 used for 3x3)
    
    # Actually for the 3x3 part, B only uses j in 0..2, k in 0..2, so V indices j*4+k for k=0,1,2
    # W for 3x3 part uses k in 0..2, i in 0..2, so W indices k*3+i
    
    # Let me define the 23 multiplications for 3x3:
    # Using a known rank-23 algorithm
    
    # I'll use the decomposition from Pan's work / standard references.
    # Terms 1-7: Strassen on A[0:2,0:2] * B[0:2,0:2]
    # Terms 8-23: Remaining 16 terms for cross terms
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10+A00)B10
    # M7 = (A01-A11)(B00+B10)
    
    # C00 = M1 + M4 - M2 - M7
    # C01 = M3 + M1
    # C10 = M2 + M5
    # C11 = M1 - M3 - M5 + M7
    
    col = 0
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add one rank-1 term. u_coeffs: dict {idx: val}, v_coeffs: dict {idx: val}, w_coeffs: dict {idx: val}"""
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    # Helper: A[i,j] -> index i*3+j in U
    # B[j,k] -> index j*4+k in V (for 3x3 part, k=0,1,2)
    # C[i,k] -> index k*3+i in W (for 3x3 part, k=0,1,2)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    add_term({0:1, 4:1}, {0:1, 5:1}, {0:1, 3:1, 4:1, 7:1})  # Wait, this isn't right
    # Actually, each term is ONE multiplication. The W should be the product's contribution to each C entry.
    # M1 contributes to C00, C01, C10, C11 with coefficients +1, +1, -1, -1
    # But in the tensor decomposition, each rank-1 term is: (sum u_i A_i)(sum v_j B_j)(sum w_k C_k)
    # So M1 = (A00+A11)(B00+B11) contributes:
    #   To C00: +1, C01: +1, C10: -1, C11: -1
    # Wait no. Let me reconsider.
    
    # In the tensor decomposition, we have:
    # T(A,B,C) = sum_r (sum_{i,j} U_{i,j,r} A_{i,j}) (sum_{j,k} V_{j,k,r} B_{j,k}) (sum_{i,k} W_{i,k,r} C_{i,k})
    
    # For Strassen, M1 = (A00+A11)(B00+B11) appears as:
    # C00 += M1, C01 += M1, C10 -= M1... no that's wrong.
    
    # Strassen:
    # C00 = M1 - M2 + M4 - M7  (wait, let me recheck)
    # C00 = M1 + M4 - M2 - M7
    # C01 = M3 + M1
    # C10 = M2 + M5  
    # C11 = M1 - M3 - M5 + M7
    
    # So M1 appears with coefficient +1 in C00, +1 in C01, 0 in C10, -1 in C11? No:
    # C00: M1 coeff = +1
    # C01: M1 coeff = +1
    # C10: M1 coeff = 0
    # C11: M1 coeff = +1
    
    # Hmm, let me just look up the exact coefficients.
    # Actually I think I'm overcomplicating. Let me just use schoolbook for now to ensure correctness,
    # then try to optimize.
    
    # For now, let me just return schoolbook rank 36 to ensure a valid submission.
    # Then in a future iteration, I'll implement the rank-32.
    
    # Reset and do schoolbook
    rank = n * m * p  # 36
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
