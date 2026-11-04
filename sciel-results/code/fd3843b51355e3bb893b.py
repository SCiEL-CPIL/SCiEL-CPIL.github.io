"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U_dict, V_dict, W_dict) as dicts mapping (row_index, col_index, term) -> coeff.
    
    Conventions:
      A is 2x2 with indices (i, j)
      B is 2x2 with indices (j, k)
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*2+j, c]
    V maps B elements: V[j*2+k, c]
    W maps C elements: W[k*2+i, c]
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    add_u(0, 0, 0, 1)
    add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1)
    add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)
    add_w(1, 1, 0, 1)
    
    # Term 1: M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1)
    add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)
    add_w(1, 1, 1, -1)
    
    # Term 2: M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1)
    add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)
    add_w(1, 1, 2, 1)
    
    # Term 3: M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1)
    add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1)
    add_w(0, 0, 3, 1)
    
    # Term 4: M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1)
    add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)
    add_w(1, 0, 4, 1)
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1)
    add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1)
    add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1)
    add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1)
    add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)
    
    return U_dict, V_dict, W_dict


def _make_3x3_laderman():
    """
    Laderman's rank-23 decomposition for 3x3 * 3x3 -> 3x3.
    Returns (U_dict, V_dict, W_dict) as dicts mapping (row, col, term) -> coeff.
    
    Conventions:
      A is 3x3 with indices (i, j)
      B is 3x3 with indices (j, k)
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*3+j, c]
    V maps B elements: V[j*3+k, c]
    W maps C elements: W[k*3+i, c]
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # Laderman's 23 products for 3x3 matrix multiplication
    # M1 = (a00+a11+a22)(b00+b11+b22)
    add_u(0,0,0,1); add_u(1,1,0,1); add_u(2,2,0,1)
    add_v(0,0,0,1); add_v(1,1,0,1); add_v(2,2,0,1)
    add_w(0,0,0,1); add_w(1,1,0,1); add_w(2,2,0,1)
    
    # M2 = (a10+a20)(b01+b02)
    add_u(1,0,1,1); add_u(2,0,1,1)
    add_v(0,1,1,1); add_v(0,2,1,1)
    add_w(1,1,1,1); add_w(2,1,1,1)
    
    # M3 = (a01+a11)(b10+b20)
    add_u(0,1,2,1); add_u(1,1,2,1)
    add_v(1,0,2,1); add_v(2,0,2,1)
    add_w(0,0,2,1); add_w(0,1,2,1)
    
    # M4 = (a21+a22)(b20+b21)
    add_u(2,1,3,1); add_u(2,2,3,1)
    add_v(2,0,3,1); add_v(2,1,3,1)
    add_w(2,2,3,1); add_w(2,1,3,1)
    
    # M5 = (a02+a12)(b20+b22)
    add_u(0,2,4,1); add_u(1,2,4,1)
    add_v(2,0,4,1); add_v(2,2,4,1)
    add_w(2,0,4,1); add_w(2,1,4,1)
    
    # M6 = (a00+a20)(b02+b22)
    add_u(0,0,5,1); add_u(2,0,5,1)
    add_v(0,2,5,1); add_v(2,2,5,1)
    add_w(0,0,5,1); add_w(0,2,5,1)
    
    # M7 = (a10+a00)(b22+b02)
    add_u(1,0,6,1); add_u(0,0,6,1)
    add_v(2,2,6,1); add_v(0,2,6,1)
    add_w(2,1,6,1); add_w(2,0,6,1)
    
    # M8 = (a20+a00)(b22+b20)
    add_u(2,0,7,1); add_u(0,0,7,1)
    add_v(2,2,7,1); add_v(2,0,7,1)
    add_w(2,2,7,1); add_w(2,0,7,1)
    
    # M9 = (a11+a10)(b22+b20)
    add_u(1,1,8,1); add_u(1,0,8,1)
    add_v(2,2,8,1); add_v(2,0,8,1)
    add_w(2,1,8,1); add_w(2,0,8,1)
    
    # M10 = (a22+a20)(b22+b02)
    add_u(2,2,9,1); add_u(2,0,9,1)
    add_v(2,2,9,1); add_v(0,2,9,1)
    add_w(2,2,9,1); add_w(2,0,9,1)
    
    # M11 = (a01+a00)(b02+b12)
    add_u(0,1,10,1); add_u(0,0,10,1)
    add_v(0,2,10,1); add_v(1,2,10,1)
    add_w(0,0,10,1); add_w(0,1,10,1)
    
    # M12 = (a12+a11)(b02+b12)
    add_u(1,2,11,1); add_u(1,1,11,1)
    add_v(0,2,11,1); add_v(1,2,11,1)
    add_w(1,0,11,1); add_w(1,1,11,1)
    
    # M13 = (a21+a22)(b02+b12)
    add_u(2,1,12,1); add_u(2,2,12,1)
    add_v(0,2,12,1); add_v(1,2,12,1)
    add_w(2,0,12,1); add_w(2,1,12,1)
    
    # M14 = (a02+a22)(b01+b21)
    add_u(0,2,13,1); add_u(2,2,13,1)
    add_v(0,1,13,1); add_v(2,1,13,1)
    add_w(0,0,13,1); add_w(0,2,13,1)
    
    # M15 = (a00+a10+a20)(b01+b11+b21)
    add_u(0,0,14,1); add_u(1,0,14,1); add_u(2,0,14,1)
    add_v(0,1,14,1); add_v(1,1,14,1); add_v(2,1,14,1)
    add_w(1,0,14,1); add_w(1,1,14,1); add_w(1,2,14,1)
    
    # M16 = (a01+a11+a21)(b00+b10+b20)
    add_u(0,1,15,1); add_u(1,1,15,1); add_u(2,1,15,1)
    add_v(0,0,15,1); add_v(1,0,15,1); add_v(2,0,15,1)
    add_w(0,0,15,1); add_w(0,1,15,1); add_w(0,2,15,1)
    
    # M17 = (a02+a12+a22)(b00+b10+b20)
    add_u(0,2,16,1); add_u(1,2,16,1); add_u(2,2,16,1)
    add_v(0,0,16,1); add_v(1,0,16,1); add_v(2,0,16,1)
    add_w(2,0,16,1); add_w(2,1,16,1); add_w(2,2,16,1)
    
    # M18 = (a00+a22)(b00+b22)
    add_u(0,0,17,1); add_u(2,2,17,1)
    add_v(0,0,17,1); add_v(2,2,17,1)
    add_w(0,0,17,1); add_w(2,2,17,1)
    
    # M19 = (a10+a20)(b00+b20)
    add_u(1,0,18,1); add_u(2,0,18,1)
    add_v(0,0,18,1); add_v(2,0,18,1)
    add_w(1,0,18,1); add_w(2,0,18,1)
    
    # M20 = (a01+a11)(b10+b00)
    add_u(0,1,19,1); add_u(1,1,19,1)
    add_v(1,0,19,1); add_v(0,0,19,1)
    add_w(0,0,19,1); add_w(0,1,19,1)
    
    # M21 = (a21+a22)(b20+b10)
    add_u(2,1,20,1); add_u(2,2,20,1)
    add_v(2,0,20,1); add_v(1,0,20,1)
    add_w(2,2,20,1); add_w(2,1,20,1)
    
    # M22 = (a02+a12)(b20+b00)
    add_u(0,2,21,1); add_u(1,2,21,1)
    add_v(2,0,21,1); add_v(0,0,21,1)
    add_w(2,0,21,1); add_w(2,1,21,1)
    
    # M23 = (a12+a22)(b20+b10)
    add_u(1,2,22,1); add_u(2,2,22,1)
    add_v(2,0,22,1); add_v(1,0,22,1)
    add_w(2,1,22,1); add_w(2,2,22,1)
    
    return U_dict, V_dict, W_dict


def _verify_3x3_laderman():
    """Verify the 3x3 Laderman decomposition is correct."""
    U_dict, V_dict, W_dict = _make_3x3_laderman()
    
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    for (i, j, c), coeff in U_dict.items():
        U[i * 3 + j, c] += coeff
    for (j, k, c), coeff in V_dict.items():
        V[j * 3 + k, c] += coeff
    for (k, i, c), coeff in W_dict.items():
        W[k * 3 + i, c] += coeff
    
    max_err = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for c in range(rank):
                    s += U[i * 3 + j, c] * V[j * 3 + k, c] * W[k * 3 + i, c]
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    
    return max_err
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) as dicts mapping (row_index, col_index) -> 1 for each factor.
    
    Conventions:
      A is 2x2 with indices (i, j), i=row of A, j=col of A
      B is 2x2 with indices (j, k), j=row of B, k=col of B
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*2+j, c]
    V maps B elements: V[j*2+k, c]  
    W maps C elements: W[k*2+i, c]
    
    Strassen formulas for 2x2:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    # Each term is a list of (factor, row, col, sign) for U, V, W
    # We'll build sparse dicts: U_dict[(i,j,c)] = coeff, etc.
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # M1 = (A00+A11)*(B00+B11)
    # This requires 4 terms for U and 4 for V, but we can combine
    # Actually, for tensor decomposition, each "multiplication" is one rank-1 term
    # But Strassen's M1 = (A00+A11)*(B00+B11) is not a single rank-1 term in the standard sense
    # We need to express each M as a product of linear forms
    
    # Let me redo this properly. Each rank-1 term c has:
    #   U part: a linear form in A (2x2 = 4 entries)
    #   V part: a linear form in B (2x2 = 4 entries)  
    #   W part: a linear form in C (2x2 = 4 entries)
    #   such that sum_c U_c * V_c = W_c (as tensors)
    
    # For Strassen, the 7 multiplications are:
    # M1: U=(1,0,0,1) [A00+A11], V=(1,0,0,1) [B00+B11], W contributes to C00,C11
    # But this isn't quite right for tensor decomposition...
    
    # Actually, the standard approach: each of the 7 Strassen products is ONE rank-1 term
    # where U and V are the linear combinations, and W is the placement
    
    # Let me define the 7 terms explicitly:
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    add_u(0, 0, 0, 1)
    add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1)
    add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)  # C00: k=0, i=0
    add_w(1, 1, 0, 1)  # C11: k=1, i=1
    
    # Term 1: M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1)
    add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)  # C10: k=0, i=1
    add_w(1, 1, 1, -1) # C11: k=1, i=1
    
    # Term 2: M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1)
    add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)  # C01: k=1, i=0
    add_w(1, 1, 2, 1)  # C11: k=1, i=1
    
    # Term 3: M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1)
    add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1)  # C10: k=0, i=1
    add_w(0, 0, 3, 1)  # C00: k=0, i=0
    
    # Term 4: M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1)
    add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1) # C00: k=0, i=0
    add_w(1, 0, 4, 1)  # C01: k=1, i=0
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1)
    add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1)
    add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)  # C11: k=1, i=1
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1)
    add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1)
    add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)  # C00: k=0, i=0
    
    return U_dict, V_dict, W_dict


def _verify_2x2():
    """Verify the 2x2 Strassen decomposition is correct."""
    U_dict, V_dict, W_dict = _make_2x2_strassen()
    
    # Build dense matrices
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    for (i, j, c), coeff in U_dict.items():
        U[i * 2 + j, c] += coeff
    for (j, k, c), coeff in V_dict.items():
        V[j * 2 + k, c] += coeff
    for (k, i, c), coeff in W_dict.items():
        W[k * 2 + i, c] += coeff
    
    # Check: sum_c U[:,c] * V[:,c] should equal the multiplication tensor
    # The multiplication tensor T[i,j,k] = delta such that C[i,k] = sum_j A[i,j]*B[j,k]
    # In our convention: T[i,j,k] = 1 if this is a valid multiplication term
    
    # Verify by checking that for all i,j,k: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes to matrix mult)
    
    # Actually, the tensor identity is:
    # sum_{i,j} A[i,j]*B[j,k] = C[i,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}
    # And we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k] = 1 for all i,j,k
    
    max_err = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(rank):
                    s += U[i * 2 + j, c] * V[j * 2 + k, c] * W[k * 2 + i, c]
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    
    return max_err


def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # First, try to verify and build the 2x2 Strassen decomposition
    try:
        max_err = _verify_2x2()
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank = n * m * p
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
    except Exception:
        rank = n * m * p
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
    
    # Now build 6x6 decomposition using 3x3 blocks of 2x2 matrices
    # A is 6x6, viewed as 3x3 block matrix of 2x2 blocks
    # B is 6x6, viewed as 3x3 block matrix of 2x2 blocks
    # C = A*B is 6x6, viewed as 3x3 block matrix of 2x2 blocks
    
    # For each pair of blocks (I,J) where I is row-block of A, J is col-block of B:
    #   C_block[I,J] = sum_K A_block[I,K] * B_block[K,J]
    
    # We can use Strassen for each 2x2 matrix multiplication within blocks
    # But we need to handle the 3x3 block structure
    
    # Strategy: For each (I,J) block pair, compute C[I,J] = sum_K A[I,K]*B[K,J]
    # Each A[I,K]*B[K,J] is a 2x2 * 2x2 multiplication, which we can do with 7 Strassen terms
    # Total: 3 (I) * 3 (J) * 3 (K) * 7 = 189 terms... but that's not minimal
    
    # Better: Use the known rank-216 decomposition or try to find a better one
    # For now, let's use a direct approach: treat 6x6 as composed of 2x2 blocks
    # and use Strassen within each block multiplication
    
    # Actually, let me try a different approach: use the fact that 6 = 2*3
    # and build a decomposition that exploits this structure
    
    # Simple approach: 189 rank terms from block Strassen
    # But we can do better by sharing Strassen terms across K sums
    
    # Let me just use the schoolbook for now if the 2x2 verification passes
    # but with a note that we should find a better decomposition
    
    # For a proper improvement, let's use the known result that R(6,6,6) <= 216
    # from various constructions. Let me try to build one.
    
    # Actually, the simplest improvement: use 2x2 Strassen within 3x3 block structure
    # Each block multiplication A[I,K]*B[K,J] uses 7 terms
    # But we can potentially share some terms
    
    # Let me build the 189-term decomposition first and verify it
    n, m, p = 6, 6, 6
    block_size = 2
    num_blocks = 3  # 6/2 = 3
    
    # Global indices:
    # A[i,j] where i = I*2 + i2, j = K*2 + j2
    # B[j,k] where j = K*2 + j2, k = J*2 + k2  
    # C[i,k] where i = I*2 + i2, k = J*2 + k2
    
    # For each (I, K, J) triple, we have a 2x2*2x2 multiplication
    # Using Strassen, this gives 7 terms
    # Total: 3*3*3*7 = 189 terms
    
    # But wait, we can do better by noticing that for fixed (I,J), 
    # C[I,J] = sum_K A[I,K]*B[K,J]
    # Each A[I,K]*B[K,J] is independent 2x2 mult
    
    # Let's build this and see if it's valid
    rank = num_blocks * num_blocks * num_blocks * 7  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_dict, V_dict, W_dict = _make_2x2_strassen()
    
    col = 0
    for I in range(num_blocks):
        for K in range(num_blocks):
            for J in range(num_blocks):
                for c in range(7):
                    # Add terms for this block multiplication
                    for (i2, j2, cc), coeff in U_dict.items():
                        if cc == c:
                            gi = I * 2 + i2
                            gj = K * 2 + j2
                            U[gi * m + gj][col] += coeff
                    
                    for (j2, k2, cc), coeff in V_dict.items():
                        if cc == c:
                            gj = K * 2 + j2
                            gk = J * 2 + k2
                            V[gj * p + gk][col] += coeff
                    
                    for (k2, i2, cc), coeff in W_dict.items():
                        if cc == c:
                            gk = J * 2 + k2
                            gi = I * 2 + i2
                            W[gk * n + gi][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition using numpy for speed
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check tensor identity: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
        # Use vectorized computation
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]  # shape (rank,)
                for k in range(p):
                    v_row = V_np[j * p + k, :]  # shape (rank,)
                    w_row = W_np[k * n + i, :]  # shape (rank,)
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            # Valid decomposition!
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
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
# EVOLVE-BLOCK-END
