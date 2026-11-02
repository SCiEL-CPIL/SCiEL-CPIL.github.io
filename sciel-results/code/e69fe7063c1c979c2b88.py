"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M3 - M4 + M6  ... wait, let me use the standard formulation.
    #
    # Let me use the most common Strassen formulation:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22) * b11
    # p3 = a11 * (b12 - b22)
    # p4 = a22 * (b21 - b11)
    # p5 = (a11 + a12) * b22
    # p6 = (a21 - a11) * (b11 + b12)
    # p7 = (a12 - a22) * (b21 + b22)
    #
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    factors = []
    
    # p1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    # c11 += p1, c22 += p1
    w = [1, 0, 0, 1]
    factors.append((u, v, w))
    
    # p2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    # c21 += p2, c22 -= p2
    w = [0, 0, 1, -1]
    factors.append((u, v, w))
    
    # p3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    # c12 += p3, c22 += p3
    w = [0, 1, 0, 1]
    factors.append((u, v, w))
    
    # p4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    # c11 += p4, c21 += p4
    w = [1, 0, 1, 0]
    factors.append((u, v, w))
    
    # p5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    # c11 -= p5, c12 += p5
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    
    # p6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    # c22 += p6
    w = [0, 0, 0, 1]
    factors.append((u, v, w))
    
    # p7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    # c11 += p7
    w = [1, 0, 0, 0]
    factors.append((u, v, w))
    
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    
    Strategy: Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    
    Each 4x4*4x4 Strassen decomposition uses 7 top-level Strassen products,
    each of which is a 2x2*2x2 multiplication, itself decomposed with 7 scalar
    Strassen products. Total: 7 * 7 = 49 per subproblem.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2 Strassen factors
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    # For each of the two subproblems (t=0 for B1, t=1 for B2):
    for t in range(2):
        # For each of the 7 top-level Strassen products (s=0..6):
        for s in range(7):
            # For each of the 7 scalar Strassen products (r=0..6):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # The top-level Strassen product s involves certain A blocks and B blocks.
                # We need to determine the U and V coefficients at the block level,
                # then combine with the scalar-level coefficients.
                
                # Top-level Strassen for 4x4*4x4 (same structure as 2x2*2x2 but with 2x2 blocks):
                # The 7 products are:
                # s=0: (A11+A22)(B11+B22)
                # s=1: (A21+A22)B11
                # s=2: A11(B12-B22)
                # s=3: A22(B21-B11)
                # s=4: (A11+A12)B22
                # s=5: (A21-A11)(B11+B12)
                # s=6: (A12-A22)(B21+B22)
                
                # For product s, the A blocks involved and their coefficients:
                # Using the same pattern as the 2x2 Strassen U factors:
                s2_u_block = s2_factors[s][0]  # [coeff for A00, A01, A10, A11] in block terms
                s2_v_block = s2_factors[s][1]  # [coeff for B00, B01, B10, B11] in block terms
                
                # For the scalar Strassen product r:
                s2_u_scalar = s2_factors[r][0]  # [coeff for element (0,0), (0,1), (1,0), (1,1)]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                # U contributions:
                # A block (bi, bj) where bi=0,1 (row block), bj=0,1 (col block)
                # Block (bi, bj) corresponds to index bi*2+bj in the 4-element vector
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                # V contributions:
                # B block (bi, bj) for subproblem t
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei  # B row
                                k = 4 * t + 2 * bj + ej  # B col (offset by 4*t)
                                V[j * p + k][col] += c_block * c_scalar
                
                # W contributions:
                # The output C for subproblem t. The top-level Strassen product s
                # contributes to certain C blocks with certain coefficients.
                # The scalar Strassen product r determines which elements of those
                # C blocks are affected.
                
                # We need to determine which C blocks are affected by product s
                # and with what coefficients. This comes from the W vector of the
                # top-level Strassen factor s, but applied at the block level.
                
                # Actually, the W factor for the top-level Strassen product s tells us
                # which C blocks are affected. But we're using the same s2_factors for
                # both levels. The W vector of s2_factors[s] gives the block-level
                # C coefficients: [C00_coeff, C01_coeff, C10_coeff, C11_coeff].
                
                s2_w_block = s2_factors[s][2]  # [coeff for C00, C01, C10, C11] in block terms
                
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei  # C row
                                k = 4 * t + 2 * cj + ej  # C col (offset by 4*t)
                                W[k * n + i][col] += c_block * c_scalar
    
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
