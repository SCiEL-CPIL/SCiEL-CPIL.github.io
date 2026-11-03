"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    factors = []
    # M1 = (A00+A11)(B00+B11)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11) * B00
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 * (B01-B11)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = A11 * (B10-B00)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # M5 = (A00+A01) * B11
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (A10-A00) * (B00+B01)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A01-A11) * (B10+B11)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Kronecker product of
    2x2x2 Strassen tensor (rank 7) with 2x2x4 tensor (rank 14 = 2*7).
    
    A(4x4) = kron(A_block[2x2], A_elem[2x2])
    B(4x8) = kron(B_block[2x2], B_elem[2x4])
    C(4x8) = kron(C_block[2x2], C_elem[2x4])
    
    Total rank = 7 * 14 = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s2 = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for t in range(2):       # sub-block of B/C (columns 4t..4t+3)
        for s in range(7):   # top-level Strassen product
            for r in range(7): # scalar Strassen product
                col = t * 49 + s * 7 + r
                
                # Block-level coefficients from Strassen product s
                ub = s2[s][0]  # A block coeffs: [A00, A01, A10, A11]
                vb = s2[s][1]  # B block coeffs: [B00, B01, B10, B11]
                wb = s2[s][2]  # C block coeffs: [C00, C01, C10, C11]
                
                # Scalar-level coefficients from Strassen product r
                us = s2[r][0]  # element coeffs within A block
                vs = s2[r][1]  # element coeffs within B block
                ws = s2[r][2]  # element coeffs within C block
                
                # U: A[i,j] where i = 2*bi+ei, j = 2*bj+ej
                for bi in range(2):
                    for bj in range(2):
                        cb = ub[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = us[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += cb * cs
                
                # V: B[j,k] where j = 2*bi+ei, k = 4*t + 2*bj+ej
                for bi in range(2):
                    for bj in range(2):
                        cb = vb[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = vs[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += cb * cs
                
                # W: C[i,k] where i = 2*ci+ei, k = 4*t + 2*cj+ej
                # W index: k*n + i (trace convention)
                for ci in range(2):
                    for cj in range(2):
                        cb = wb[ci * 2 + cj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = ws[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += cb * cs
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition (rank 98) first, falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    
    # Try Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
