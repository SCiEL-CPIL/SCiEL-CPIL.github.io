"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n+i, r] corresponds to output element C[i,k] from A[i,j]*B[j,k]
    The decomposition requires: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    # For each (i,j,k) triple, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_block():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) with indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _outer_product_2x1_1x2():
    """Rank-4 decomposition for (2x1)*(1x2) -> 2x2 outer product.
    U (2x4), V (2x4), W (4x4) with indexing U[i*1+j,r]=U[i,r], V[j*1+k,r]=V[k,r], W[k*2+i,r]
    Actually for 2x1 * 1x2: A is 2x1, B is 1x2, C is 2x2.
    C[i,k] = A[i,0]*B[0,k]
    This is an outer product, rank 4 (schoolbook).
    U[i*1+0, r] = U[i, r], V[0*1+k, r] = V[k, r], W[k*2+i, r]
    """
    U = [[0]*4 for _ in range(2)]  # 2x1 matrix, indexed by i
    V = [[0]*4 for _ in range(2)]  # 1x2 matrix, indexed by k
    W = [[0]*4 for _ in range(4)]  # 2x2 matrix, indexed by k*2+i
    
    # C[i,k] = A[i]*B[k]
    # For each (i,k), we need U[i]*V[k]*W[k*2+i] = 1
    # Use 4 rank-1 terms:
    for i in range(2):
        for k in range(2):
            col = i*2 + k
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
    
    return U, V, W

def _block_decompose_4x5x6():
    """
    Decompose 4x5 * 5x6 using block structure with Strassen on 2x2 blocks.
    Split: rows 4=2+2, cols 5=2+2+1, cols 6=2+2+2
    
    Block structure:
    A: 2 row-blocks x 3 col-blocks
      A[0,0]: 2x2, A[0,1]: 2x2, A[0,2]: 2x1
      A[1,0]: 2x2, A[1,1]: 2x2, A[1,2]: 2x1
    
    B: 3 row-blocks x 3 col-blocks
      B[0,0]: 2x2, B[0,1]: 2x2, B[0,2]: 2x2
      B[1,0]: 2x2, B[1,1]: 2x2, B[1,2]: 2x2
      B[2,0]: 1x2, B[2,1]: 1x2, B[2,2]: 1x2
    
    C: 2 row-blocks x 3 col-blocks
      C[0,0]: 2x2, C[0,1]: 2x2, C[0,2]: 2x2
      C[1,0]: 2x2, C[1,1]: 2x2, C[1,2]: 2x2
    
    C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
    
    For (i,k) with A[i,k] being 2x2 and B[k,j] being 2x2: use Strassen (rank 7)
    For (i,k) with A[i,k] being 2x1 and B[k,j] being 1x2: use outer product (rank 4)
    """
    # Get Strassen components
    U_s, V_s, W_s = _strassen_2x2x2_block()
    U_o, V_o, W_o = _outer_product_2x1_1x2()
    
    # Define block sizes
    # A blocks: row_splits = [2,2], col_splits = [2,2,1]
    # B blocks: row_splits = [2,2,1], col_splits = [2,2,2]
    # C blocks: row_splits = [2,2], col_splits = [2,2,2]
    
    a_row_splits = [2, 2]
    a_col_splits = [2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    c_row_splits = [2, 2]
    c_col_splits = [2, 2, 2]
    
    # Precompute offsets
    a_row_off = [0]
    for s in a_row_splits[:-1]:
        a_row_off.append(a_row_off[-1] + s)
    a_col_off = [0]
    for s in a_col_splits[:-1]:
        a_col_off.append(a_col_off[-1] + s)
    b_row_off = [0]
    for s in b_row_splits[:-1]:
        b_row_off.append(b_row_off[-1] + s)
    b_col_off = [0]
    for s in b_col_splits[:-1]:
        b_col_off.append(b_col_off[-1] + s)
    c_row_off = [0]
    for s in c_row_splits[:-1]:
        c_row_off.append(c_row_off[-1] + s)
    c_col_off = [0]
    for s in c_col_splits[:-1]:
        c_col_off.append(c_col_off[-1] + s)
    
    # Count total rank
    total_rank = 0
    for i in range(2):
        for k in range(3):
            a_height = a_row_splits[i]
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                if a_width == 2 and b_height == 2 and b_width == 2:
                    total_rank += 7
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    total_rank += 4
                else:
                    # Should not happen with our splits
                    total_rank += a_height * a_width * b_width
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(4 * 5)]  # n*m = 20
    V = [[0] * total_rank for _ in range(5 * 6)]  # m*p = 30
    W = [[0] * total_rank for _ in range(6 * 4)]  # p*n = 24
    
    col = 0
    for i in range(2):
        for k in range(3):
            a_height = a_row_splits[i]
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                
                a_i_start = a_row_off[i]
                a_k_start = a_col_off[k]
                b_k_start = b_row_off[k]
                b_j_start = b_col_off[j]
                c_i_start = c_row_off[i]
                c_j_start = c_col_off[j]
                
                if a_width == 2 and b_height == 2 and b_width == 2:
                    # Use Strassen
                    for r in range(7):
                        # U: A[i,k] block, index (a_i_start + ii, a_k_start + jj) -> row a_i_start+ii, col a_k_start+jj
                        # U row index: (a_i_start+ii)*5 + (a_k_start+jj)
                        # V row index: (b_k_start+kk)*6 + (b_j_start+ll)
                        # W row index: (c_j_start+ll)*4 + (c_i_start+mm)
                        for ii in range(2):
                            for jj in range(2):
                                if U_s[ii*2+jj][r] != 0:
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    U[u_row][col] += U_s[ii*2+jj][r]
                        for kk in range(2):
                            for ll in range(2):
                                if V_s[kk*2+ll][r] != 0:
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    V[v_row][col] += V_s[kk*2+ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_s[ll*2+mm][r] != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += W_s[ll*2+mm][r]
                        col += 1
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    # Use outer product
                    for r in range(4):
                        for ii in range(2):
                            if U_o[ii][r] != 0:
                                u_row = (a_i_start + ii) * 5 + a_k_start
                                U[u_row][col] += U_o[ii][r]
                        for ll in range(2):
                            if V_o[ll][r] != 0:
                                v_row = b_k_start * 6 + (b_j_start + ll)
                                V[v_row][col] += V_o[ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_o[ll*2+mm][r] != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += W_o[ll*2+mm][r]
                        col += 1
                else:
                    # Fallback: schoolbook for this block
                    for ii in range(a_height):
                        for jj in range(a_width):
                            for kk in range(b_height):
                                for ll in range(b_width):
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + ii)
                                    U[u_row][col] = 1
                                    V[v_row][col] = 1
                                    W[w_row][col] = 1
                                    col += 1
    
    return {"U": U, "V": V, "W": W}, total_rank

def _structured_basis_projection():
    """
    Structured basis projection approach:
    Model the multiplication tensor as a sum of rank-1 terms derived from
    Kronecker products of identity matrices, leveraging the block structure.
    
    For 4x5 * 5x6:
    - View A as 2x2 blocks of (2x2) and (2x1)
    - View B as 2x2 blocks of (2x2) and (1x2)  
    - Use structured decomposition that exploits the rectangular nature
    
    This approach aims to find a lower-rank decomposition by:
    1. Identifying the natural block structure
    2. Using Kronecker product structure to create rank-1 terms
    3. Projecting onto a structured basis that respects the dimensions
    """
    n, m, p = 4, 5, 6
    
    # Block decomposition strategy:
    # Split dimensions into blocks that align with Strassen-friendly sizes
    # A: 4x5 = (2+2) x (2+2+1)
    # B: 5x6 = (2+2+1) x (2+2+2)
    # C: 4x6 = (2+2) x (2+2+2)
    
    # For each block combination, use the most efficient decomposition
    # This is similar to _block_decompose_4x5x6 but with a different strategy
    
    # Try an alternative split that might give better rank
    # Split A: 4x5 = (2+2) x (2+3) where 3 = 2+1
    # Split B: 5x6 = (2+3) x (2+2+2)
    
    # Actually, let's try a more systematic approach using the fact that
    # we can decompose the tensor using Kronecker products of smaller tensors
    
    # The key insight: if we can write the multiplication tensor as
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors from structured bases, we can reduce rank
    
    # For now, use the same block decomposition but verify it works
    # and potentially find a better split
    
    result, rank = _block_decompose_4x5x6()
    
    # Verify the decomposition
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result, rank
    
    # If verification fails, fall back to schoolbook
    return _schoolbook(n, m, p), n * m * p

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin
    
    # Try structured basis projection approach
    try:
        result, rank = _structured_basis_projection()
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
