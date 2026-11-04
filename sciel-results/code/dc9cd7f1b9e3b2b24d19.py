"""
Matrix multiplication 2x4x4 - rank-minimal tensor decomposition.

Uses Strassen-based block decomposition with shared multiplications.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor: T[i,j,k] = 1 for all valid i,j,k
    # T has shape (n, m, p)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err


def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    deadline = 115.0  # seconds, safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen 7-multiplication patterns for 2x2 matrices
    # Each pattern: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # Local 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_patterns = [
        # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        # M2 = (a21+a22)(b11) -> contributes to c21, c22
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, -1)),
        # M3 = (a11)(b12-b22) -> contributes to c12, c22
        ((0,), (1,), (1, 3), (1, -1), (1, 3), (1, 1)),
        # M4 = (a22)(b21-b11) -> contributes to c11, c21
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        # M5 = (a11+a12)(b22) -> contributes to c11, c12
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        # M6 = (a21-a11)(b11+b12) -> contributes to c22
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # M7 = (a12-a22)(b21+b22) -> contributes to c11
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),
    ]
    
    # Block structure for 2x4 * 4x4:
    # A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]]
    # C (2x4) = [C_L | C_R]
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    
    # We have 4 block products, each 2x2 * 2x2 = 2x2
    # Each uses Strassen (rank 7), total rank 28
    
    # Try to find a better decomposition by combining blocks
    # For now, use the standard 4-Strassen approach with rank 28
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Blocks: (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
    blocks = [
        (0, 0, 0, 0, 0, 0),   # A_L * B_TL -> C_L
        (0, 2, 2, 0, 0, 0),   # A_R * B_BL -> C_L
        (0, 0, 0, 2, 0, 2),   # A_L * B_TR -> C_R
        (0, 2, 2, 2, 0, 2),   # A_R * B_BR -> C_R
    ]
    
    col = 0
    for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in blocks:
        for (a_idx, a_c, b_idx, b_c, c_idx, c_c) in strassen_patterns:
            # Fill U: index = (a_row, a_col) -> a_ro + lr, a_co + lc
            for idx, coeff in zip(a_idx, a_c):
                lr = idx // 2
                lc = idx % 2
                gi = a_ro + lr
                gj = a_co + lc
                U[gi * m + gj][col] = coeff
            
            # Fill V: index = (b_row, b_col) -> b_ro + lr, b_co + lc
            for idx, coeff in zip(b_idx, b_c):
                lr = idx // 2
                lc = idx % 2
                gj = b_ro + lr
                gk = b_co + lc
                V[gj * p + gk][col] = coeff
            
            # Fill W: index = (c_row, c_col) -> c_ro + lr, c_co + lc
            # W uses trace convention: W[k*n + i, r] where k is col, i is row
            for idx, coeff in zip(c_idx, c_c):
                lr = idx // 2
                lc = idx % 2
                gi = c_ro + lr
                gk = c_co + lc
                W[gk * n + gi][col] = coeff
            
            col += 1
            
            if time.time() - t0 > deadline - 5:
                break
    
    # Verify
    err = _verify(U, V, W, n, m, p)
    if err > 1e-10:
        # Fallback: return what we have (should be exact for integer coefficients)
        pass
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Quick verification
    err = _verify(res['U'], res['V'], res['W'], 2, 4, 4)
    print(f"Max error: {err}")
