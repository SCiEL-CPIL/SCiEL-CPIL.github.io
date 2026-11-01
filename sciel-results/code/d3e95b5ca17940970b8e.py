# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] where C[k*n+i] = sum_j A[i,j] * B[j,k]
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Compute actual tensor from decomposition
    T_actual = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = np.abs(T_actual - T_expected)
    max_diff = np.max(diff)
    return max_diff < tol, max_diff

def _try_rank12(deadline):
    """
    Attempt to find a rank-12 decomposition using a structured algebraic approach.
    The idea is to exploit the rectangular structure more aggressively.
    """
    n, m, p = 2, 2, 4
    rank = 12
    
    # Try a specific rank-12 structure based on Laderman-style extensions
    # This is a heuristic attempt; if it fails, we fall back to rank-14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Try a different set of bilinear forms
    # T1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # T2 = (A10+A11)(B00+B11)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[5][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # T3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # T4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # T5 = (A00+A01)(B02+B13)
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1; V[7][4] = 1
    W[4][4] = 1; W[7][4] = 1
    
    # T6 = (A10+A11)(B02+B13)
    U[2][5] = 1; U[3][5] = 1
    V[2][5] = 1; V[7][5] = 1
    W[5][5] = 1; W[7][5] = 1
    
    # T7 = A00(B03-B13)
    U[0][6] = 1
    V[3][6] = 1; V[7][6] = -1
    W[6][6] = 1; W[7][6] = 1
    
    # T8 = A11(B12-B02)
    U[3][7] = 1
    V[6][7] = 1; V[2][7] = -1
    W[4][7] = 1; W[5][7] = 1
    
    # T9 = (A01-A11)(B10+B11)
    U[1][8] = 1; U[3][8] = -1
    V[4][8] = 1; V[5][8] = 1
    W[0][8] = 1
    
    # T10 = (A10-A00)(B00+B01)
    U[2][9] = 1; U[0][9] = -1
    V[0][9] = 1; V[1][9] = 1
    W[3][9] = 1
    
    # T11 = (A01-A11)(B12+B13)
    U[1][10] = 1; U[3][10] = -1
    V[6][10] = 1; V[7][10] = 1
    W[4][10] = 1
    
    # T12 = (A10-A00)(B02+B03)
    U[2][11] = 1; U[0][11] = -1
    V[2][11] = 1; V[3][11] = 1
    W[7][11] = 1
    
    result = {"U": U, "V": V, "W": W}
    valid, max_diff = _verify_decomposition(result)
    if valid:
        return result
    
    return None

def construct():
    deadline = time.time() + 120  # Safety margin
    
    # Try rank-12 first
    result12 = _try_rank12(deadline)
    if result12 is not None:
        return result12
    
    # Fall back to rank-14
    return _build_rank14()
# EVOLVE-BLOCK-END
