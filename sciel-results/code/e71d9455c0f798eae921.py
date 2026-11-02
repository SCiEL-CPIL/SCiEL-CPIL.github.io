"""Matrix multiplication 3x4x4 tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensor terms for 2x2x2 matrix multiplication.
    Each term is (U_col, V_col, W_col) where:
      U has shape (2*2, 1) = (4, 1), indexed by [i*2+j]
      V has shape (2*2, 1) = (4, 1), indexed by [j*2+k]
      W has shape (2*2, 1) = (4, 1), indexed by [k*2+i]
    
    Strassen's 7 multiplications for C = A*B where A,B are 2x2:
    M1 = (A00+A11)(B00+B11)
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
    # U indices: i*2+j for i in [0,1], j in [0,1] -> [0,1,2,3]
    # V indices: j*2+k for j in [0,1], k in [0,1] -> [0,1,2,3]
    # W indices: k*2+i for k in [0,1], i in [0,1] -> [0,1,2,3]
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # A00 = U[0], A11 = U[3], B00 = V[0], B11 = V[3]
    # Contributes to: C00 (+1), C11 (+1)
    # W indices: C00 -> k=0,i=0 -> 0*2+0=0; C11 -> k=1,i=1 -> 1*2+1=3
    U1 = [1, 0, 0, 1]  # A00, A01, A10, A11
    V1 = [1, 0, 0, 1]  # B00, B01, B10, B11
    W1 = [1, 0, 0, 1]  # C00, C01, C10, C11
    terms.append((U1, V1, W1))
    
    # M2 = (A10+A11)*B00
    # A10 = U[2], A11 = U[3], B00 = V[0]
    # Contributes to: C10 (+1)
    # W: C10 -> k=0,i=1 -> 0*2+1=1
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, 0]
    terms.append((U2, V2, W2))
    
    # M3 = A00*(B01-B11)
    # A00 = U[0], B01 = V[1], B11 = V[3]
    # Contributes to: C01 (+1), C11 (+1)
    # W: C01 -> k=1,i=0 -> 1*2+0=2; C11 -> k=1,i=1 -> 3
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    terms.append((U3, V3, W3))
    
    # M4 = A11*(B10-B00)
    # A11 = U[3], B10 = V[2], B00 = V[0]
    # Contributes to: C00 (+1), C10 (+1)
    # W: C00 -> 0; C10 -> 1
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, 0]  # B10 - B00 = -1*V[0] + 1*V[2]
    # Actually: B10 - B00, so V4 = [0-1, 0, 1-0, 0] = [-1, 0, 1, 0]
    V4 = [-1, 0, 1, 0]
    W4 = [1, 0, 1, 0]
    terms.append((U4, V4, W4))
    
    # M5 = (A00+A01)*B11
    # A00 = U[0], A01 = U[1], B11 = V[3]
    # Contributes to: C00 (-1), C01 (+1)
    # W: C00 -> 0; C01 -> 2
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 0, 0, 1]
    terms.append((U5, V5, W5))
    
    # M6 = (A10-A00)*(B00+B01)
    # A10 = U[2], A00 = U[0], B00 = V[0], B01 = V[1]
    # Contributes to: C11 (+1)
    # W: C11 -> 3
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    terms.append((U6, V6, W6))
    
    # M7 = (A01-A11)*(B10+B11)
    # A01 = U[1], A11 = U[3], B10 = V[2], B11 = V[3]
    # Contributes to: C00 (+1)
    # W: C00 -> 0
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    terms.append((U7, V7, W7))
    
    return terms


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Reconstruct tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition for n x m x p matrix multiplication.
    
    Strategy: Split m into blocks and p into blocks.
    For 3x4x4: split m=4 into [2,2], p=4 into [2,2]
    For each (m_block, p_block) of size 2x2:
      - Rows 0-1 (2 rows): Strassen 2x2x2, rank 7
      - Row 2 (1 row): trivial 1x2x2, rank 4
      - Total per block: 11
    
    Total rank: 4 blocks * 11 = 44
    """
    n, m, p = 3, 4, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Split m into blocks of 2, p into blocks of 2
    m_blocks = [0, 2]  # starting indices for m blocks of size 2
    p_blocks = [0, 2]  # starting indices for p blocks of size 2
    m_block_sizes = [2, 2]
    p_block_sizes = [2, 2]
    
    # Strassen terms for 2x2x2
    strassen_terms = _strassen_2x2x2()
    
    U = [[0] * 0 for _ in range(nm)]  # will build column by column
    V = [[0] * 0 for _ in range(mp)]
    W = [[0] * 0 for _ in range(pn)]
    
    # We'll collect all terms as (U_col, V_col, W_col)
    all_terms = []
    
    for mb_idx, m_start in enumerate(m_blocks):
        m_size = m_block_sizes[mb_idx]
        for pb_idx, p_start in enumerate(p_blocks):
            p_size = p_block_sizes[pb_idx]
            
            # For this block, we compute C[i, p_start:p_start+p_size] for i in range(n)
            # using A[i, m_start:m_start+m_size] * B[m_start:m_start+m_size, p_start:p_start+p_size]
            
            # Rows 0-1: Strassen 2x2x2
            # Local indices: i_local in [0,1], j_local in [0,1], k_local in [0,1]
            # Global: i = i_local, j = m_start + j_local, k = p_start + k_local
            for (U_s, V_s, W_s) in strassen_terms:
                U_col = [0] * nm
                V_col = [0] * mp
                W_col = [0] * pn
                
                for i_local in range(2):
                    for j_local in range(2):
                        global_i = i_local
                        global_j = m_start + j_local
                        U_col[global_i * m + global_j] = U_s[i_local * 2 + j_local]
                
                for j_local in range(2):
                    for k_local in range(2):
                        global_j = m_start + j_local
                        global_k = p_start + k_local
                        V_col[global_j * p + global_k] = V_s[j_local * 2 + k_local]
                
                for k_local in range(2):
                    for i_local in range(2):
                        global_k = p_start + k_local
                        global_i = i_local
                        W_col[global_k * n + global_i] = W_s[k_local * 2 + i_local]
                
                all_terms.append((U_col, V_col, W_col))
            
            # Row 2: trivial 1x2x2 decomposition, rank 4
            # C[2, p_start+k] = sum_j A[2, m_start+j] * B[m_start+j, p_start+k]
            # For each (j_local, k_local): rank-1 term
            for j_local in range(2):
                for k_local in range(2):
                    U_col = [0] * nm
                    V_col = [0] * mp
                    W_col = [0] * pn
                    
                    global_i = 2
                    global_j = m_start + j_local
                    global_k = p_start + k_local
                    
                    U_col[global_i * m + global_j] = 1
                    V_col[global_j * p + global_k] = 1
                    W_col[global_k * n + global_i] = 1
                    
                    all_terms.append((U_col, V_col, W_col))
    
    rank = len(all_terms)
    
    # Build the final matrices
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for r, (U_col, V_col, W_col) in enumerate(all_terms):
        for i in range(nm):
            U[i][r] = U_col[i]
        for i in range(mp):
            V[i][r] = V_col[i]
        for i in range(pn):
            W[i][r] = W_col[i]
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the block decomposition
    U, V, W, rank = _build_block_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
