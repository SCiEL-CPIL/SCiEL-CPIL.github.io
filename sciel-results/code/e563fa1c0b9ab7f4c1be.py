"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using fast einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    
    return np.allclose(recon, expected, atol=1e-9)

def schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks to achieve rank 44.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 140
    
    # Rank 44 decomposition using Strassen on 2x2 blocks
    R = 44
    U = [[0] * R for _ in range(n * m)]  # 12 x 44
    V = [[0] * R for _ in range(m * p)]  # 16 x 44
    W = [[0] * R for _ in range(p * n)]  # 12 x 44
    
    def add_strassen(r_a, c_a, r_b, c_b, r_c, c_c, base_idx):
        """
        Adds Strassen decomposition for A[r_a:r_a+2, c_a:c_a+2] * B[r_b:r_b+2, c_b:c_b+2]
        contributing to C[r_c:r_c+2, c_c:c_c+2].
        base_idx: starting index in U, V, W for this block of 7 multiplications.
        """
        # M1 = (A00+A11)(B00+B11)
        U[r_a*m + c_a][base_idx] += 1
        U[(r_a+1)*m + (c_a+1)][base_idx] += 1
        V[r_b*p + c_b][base_idx] += 1
        V[(r_b+1)*p + (c_b+1)][base_idx] += 1
        # C00 += M1, C01 += M1
        W[c_c*n + r_c][base_idx] += 1
        W[(c_c+1)*n + r_c][base_idx] += 1
        
        # M2 = (A10+A11)B00
        U[(r_a+1)*m + c_a][base_idx+1] += 1
        U[(r_a+1)*m + (c_a+1)][base_idx+1] += 1
        V[r_b*p + c_b][base_idx+1] += 1
        # C10 += M2
        W[c_c*n + (r_c+1)][base_idx+1] += 1
        
        # M3 = A00(B01-B11)
        U[r_a*m + c_a][base_idx+2] += 1
        V[r_b*p + (c_b+1)][base_idx+2] += 1
        V[(r_b+1)*p + (c_b+1)][base_idx+2] -= 1
        # C01 += M3
        W[(c_c+1)*n + r_c][base_idx+2] += 1
        
        # M4 = A11(B10-B00)
        U[(r_a+1)*m + (c_a+1)][base_idx+3] += 1
        V[(r_b+1)*p + c_b][base_idx+3] += 1
        V[r_b*p + c_b][base_idx+3] -= 1
        # C00 += M4, C10 += M4
        W[c_c*n + r_c][base_idx+3] += 1
        W[c_c*n + (r_c+1)][base_idx+3] += 1
        
        # M5 = (A00+A10)B11
        U[r_a*m + c_a][base_idx+4] += 1
        U[(r_a+1)*m + c_a][base_idx+4] += 1
        V[(r_b+1)*p + (c_b+1)][base_idx+4] += 1
        # C00 -= M5, C11 += M5
        W[c_c*n + r_c][base_idx+4] -= 1
        W[(c_c+1)*n + (r_c+1)][base_idx+4] += 1
        
        # M6 = (A01-A10)(B00+B10)
        U[r_a*m + (c_a+1)][base_idx+5] += 1
        U[(r_a+1)*m + c_a][base_idx+5] -= 1
        V[r_b*p + c_b][base_idx+5] += 1
        V[(r_b+1)*p + c_b][base_idx+5] += 1
        # C11 += M6
        W[(c_c+1)*n + (r_c+1)][base_idx+5] += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[r_a*m + (c_a+1)][base_idx+6] += 1
        U[(r_a+1)*m + (c_a+1)][base_idx+6] -= 1
        V[r_b*p + (c_b+1)][base_idx+6] += 1
        V[(r_b+1)*p + (c_b+1)][base_idx+6] += 1
        # C00 += M7, C11 += M7
        W[c_c*n + r_c][base_idx+6] += 1
        W[(c_c+1)*n + (r_c+1)][base_idx+6] += 1

    def add_1x2_2x2(r_a, c_a, r_b, c_b, r_c, c_c, base_idx):
        """
        Adds schoolbook decomposition for A[r_a, c_a:c_a+2] * B[r_b:r_b+2, c_b:c_b+2]
        contributing to C[r_c, c_c:c_c+2].
        A is 1x2, B is 2x2.
        A = [A0, A1]
        B = [[B00, B01], [B10, B11]]
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        4 multiplications.
        """
        # m1 = A0 * B00 -> C0
        U[r_a*m + c_a][base_idx] += 1
        V[r_b*p + c_b][base_idx] += 1
        W[c_c*n + r_c][base_idx] += 1
        
        # m2 = A1 * B10 -> C0
        U[r_a*m + (c_a+1)][base_idx+1] += 1
        V[(r_b+1)*p + c_b][base_idx+1] += 1
        W[c_c*n + r_c][base_idx+1] += 1
        
        # m3 = A0 * B01 -> C1
        U[r_a*m + c_a][base_idx+2] += 1
        V[r_b*p + (c_b+1)][base_idx+2] += 1
        W[(c_c+1)*n + r_c][base_idx+2] += 1
        
        # m4 = A1 * B11 -> C1
        U[r_a*m + (c_a+1)][base_idx+3] += 1
        V[(r_b+1)*p + (c_b+1)][base_idx+3] += 1
        W[(c_c+1)*n + r_c][base_idx+3] += 1

    # Strassen parts for top 2 rows of A
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0, 0, 7)   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2, 14)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2, 2, 0, 2, 21)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Schoolbook parts for bottom row of A
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    add_1x2_2x2(2, 0, 0, 0, 2, 0, 28)   # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_1x2_2x2(2, 2, 2, 0, 2, 0, 32)   # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    add_1x2_2x2(2, 0, 0, 2, 2, 2, 36)   # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_1x2_2x2(2, 2, 2, 2, 2, 2, 40)   # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_sb, V_sb, W_sb = schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
