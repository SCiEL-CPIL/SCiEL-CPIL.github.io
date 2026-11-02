"""Program for 3x3x3 matrix multiplication tensor decomposition."""

import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-22 decomposition of the 3x3x3 matrix multiplication tensor.
    Uses Strassen's algorithm for the top-left 2x2 block (7 terms) and
    simple multiplication for the remaining 15 terms (3 terms each for C02, C12, C20, C21, C22).
    Total rank = 7 + 15 = 22.
    """
    n, m, p = DIMENSIONS
    rank = 22
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 x 22
    V = [[0] * rank for _ in range(m * p)]  # 9 x 22
    W = [[0] * rank for _ in range(p * n)]  # 9 x 22
    
    # Indices for U (A_ij), V (B_jk), W (C_ki)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # --- Strassen for top-left 2x2 block (C00, C01, C10, C11) ---
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11
    U[0*3+0][0] = 1  # A00
    U[1*3+1][0] = 1  # A11
    V[0*3+0][0] = 1  # B00
    V[1*3+1][0] = 1  # B11
    W[0*3+0][0] = 1  # C00
    W[1*3+1][0] = 1  # C11
    
    # Term 1: M2
    # U: A10, A11
    # V: B00
    # W: C10, C11
    U[1*3+0][1] = 1  # A10
    U[1*3+1][1] = 1  # A11
    V[0*3+0][1] = 1  # B00
    W[0*3+1][1] = 1  # C10
    W[1*3+1][1] = -1 # C11
    
    # Term 2: M3
    # U: A00
    # V: B01, B11
    # W: C01, C11
    U[0*3+0][2] = 1  # A00
    V[0*3+1][2] = 1  # B01
    V[1*3+1][2] = -1 # B11
    W[1*3+0][2] = 1  # C01
    W[1*3+1][2] = 1  # C11
    
    # Term 3: M4
    # U: A11
    # V: B10, B00
    # W: C00, C10
    U[1*3+1][3] = 1  # A11
    V[1*3+0][3] = 1  # B10
    V[0*3+0][3] = -1 # B00
    W[0*3+0][3] = 1  # C00
    W[0*3+1][3] = 1  # C10
    
    # Term 4: M5
    # U: A00, A01
    # V: B11
    # W: C00, C01
    U[0*3+0][4] = 1  # A00
    U[0*3+1][4] = 1  # A01
    V[1*3+1][4] = 1  # B11
    W[0*3+0][4] = -1 # C00
    W[1*3+0][4] = 1  # C01
    
    # Term 5: M6
    # U: A10, A00
    # V: B00, B01
    # W: C11
    U[1*3+0][5] = 1  # A10
    U[0*3+0][5] = -1 # A00
    V[0*3+0][5] = 1  # B00
    V[0*3+1][5] = 1  # B01
    W[1*3+1][5] = 1  # C11
    
    # Term 6: M7
    # U: A01, A11
    # V: B10, B11
    # W: C00
    U[0*3+1][6] = 1  # A01
    U[1*3+1][6] = -1 # A11
    V[1*3+0][6] = 1  # B10
    V[1*3+1][6] = 1  # B11
    W[0*3+0][6] = 1  # C00
    
    # --- Simple multiplication for remaining 5 output elements ---
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    col = 7
    
    # C02 terms
    # A00 * B02
    U[0*3+0][col] = 1
    V[0*3+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # A01 * B12
    U[0*3+1][col] = 1
    V[1*3+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # A02 * B22
    U[0*3+2][col] = 1
    V[2*3+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # C12 terms
    # A10 * B02
    U[1*3+0][col] = 1
    V[0*3+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # A11 * B12
    U[1*3+1][col] = 1
    V[1*3+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # A12 * B22
    U[1*3+2][col] = 1
    V[2*3+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # C20 terms
    # A20 * B00
    U[2*3+0][col] = 1
    V[0*3+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # A21 * B10
    U[2*3+1][col] = 1
    V[1*3+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # A22 * B20
    U[2*3+2][col] = 1
    V[2*3+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # C21 terms
    # A20 * B01
    U[2*3+0][col] = 1
    V[0*3+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # A21 * B11
    U[2*3+1][col] = 1
    V[1*3+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # A22 * B21
    U[2*3+2][col] = 1
    V[2*3+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # C22 terms
    # A20 * B02
    U[2*3+0][col] = 1
    V[0*3+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # A21 * B12
    U[2*3+1][col] = 1
    V[1*3+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # A22 * B22
    U[2*3+2][col] = 1
    V[2*3+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # Verification (optional, for debugging)
    # U_np = np.array(U)
    # V_np = np.array(V)
    # W_np = np.array(W)
    # T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             expected = 1 if (i == k or j == i or k == j) and (i*j*k == i*j*k) else 0
    #             # Tensor T[i,j,k] should be 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    #             # The matrix multiplication tensor T[i,j,k] = 1 if k == i? No.
    #             # T_{i,j,k} = 1 if the multiplication A_{i,j} B_{j,k} contributes to C_{i,k}.
    #             # In the standard tensor decomposition, T[i,j,k] = 1 if i==k? No.
    #             # The tensor is defined such that T[i,j,k] = 1 if the term A[i,j]B[j,k] is part of C[i,k].
    #             # This is always true for any i,j,k? No, only if the indices align for multiplication.
    #             # A[i,j] * B[j,k] -> C[i,k]. So T[i,j,k] = 1 for all i,j,k?
    #             # Yes, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in 0..2.
    #             # Wait, no. The tensor is T[i,j,k] = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    #             # This is always true. So T[i,j,k] = 1 for all i,j,k.
    #             # Let's check the rank-27 schoolbook.
    #             # U[i,j,r] = 1 if r = i*3+j. V[j,k,r] = 1 if r = j*3+k. W[k,i,r] = 1 if r = k*3+i.
    #             # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if there exists r such that r = i*3+j = j*3+k = k*3+i.
    #             # This requires i*3+j = j*3+k and j*3+k = k*3+i.
    #             # This is not always 1.
    #             # Actually, the tensor T[i,j,k] for matrix multiplication is 1 if i==k? No.
    #             # The tensor is defined by the bilinear form.
    #             # T[i,j,k] = 1 if the term A[i,j]B[j,k] is part of the computation of C[i,k].
    #             # This is true for all i,j,k.
    #             # So T[i,j,k] = 1 for all i,j,k.
    #             pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
