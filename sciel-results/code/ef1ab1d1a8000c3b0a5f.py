"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and k == k2) else 0.0
                            actual = T[a, b, c]
                            if abs(actual - expected) > 1e-9:
                                errors += 1
                                if errors > 10:
                                    return False
    return errors == 0

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2():
    """
    Return U, V, W factor matrices for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    """
    factors = [
        ([0,0,0,1],[0,0,0,1],[0,0,0,1]),  # M1
        ([0,0,1,1],[1,0,0,0],[0,0,1,1]),  # M2
        ([1,0,0,0],[0,0,1,-1],[0,1,0,0]),  # M3
        ([0,0,0,1],[1,-1,0,0],[1,0,1,0]),  # M4
        ([1,1,0,0],[0,0,0,1],[1,1,0,0]),  # M5
        ([1,0,-1,0],[1,1,0,0],[0,0,0,1]),  # M6
        ([0,1,0,-1],[0,0,1,1],[1,0,0,0])   # M7
    ]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for r, (u, v, w) in enumerate(factors):
        for idx in range(4):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
            
    return U, V, W

def _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col):
    """
    Add Strassen 2x2 decomposition for a block at specific offsets.
    u_r0, u_c0: top-left row/col of A block in global 5x5.
    v_r0, v_c0: top-left row/col of B block in global 5x5.
    w_r0, w_c0: top-left row/col of C block in global 5x5.
    """
    sU, sV, sW = _strassen_2x2()
    
    for r in range(7):
        for local_idx in range(4):
            li = local_idx // 2
            lj = local_idx % 2
            
            # Map local 2x2 index to global index
            # A block: rows u_r0..u_r0+1, cols u_c0..u_c0+1
            u_global = (u_r0 + li) * 5 + (u_c0 + lj)
            
            # B block: rows v_r0..v_r0+1, cols v_c0..v_c0+1
            # Note: B index is j'*p + k. Here p=5.
            # Local B index li, lj corresponds to row li, col lj in the 2x2 block.
            v_global = (v_r0 + li) * 5 + (v_c0 + lj)
            
            # C block: rows w_r0..w_r0+1, cols w_c0..w_c0+1
            # Note: C index is k'*n + i'. Here n=5.
            # Local C index li, lj corresponds to row li, col lj in the 2x2 block.
            # Wait, C index is k'*5 + i'.
            # In the 2x2 block, local row is k', local col is i'.
            w_global = (w_r0 + li) * 5 + (w_c0 + lj)
            
            U[u_global][col + r] += sU[local_idx][r]
            V[v_global][col + r] += sV[local_idx][r]
            W[w_global][col + r] += sW[local_idx][r]
            
    return col + 7

def _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col):
    """
    Add schoolbook decomposition for a rectangular block multiplication.
    u_rows, u_cols: list of global row indices for A block, list of global col indices for A block?
    Actually, it's easier to pass ranges or lists of global indices for the specific elements.
    Let's pass lists of global indices for A, B, C elements that participate.
    But schoolbook is term-by-term.
    A[i,j] * B[j,k] -> C[i,k].
    We need to iterate i in A_rows, j in A_cols (which is B_rows), k in B_cols (which is C_cols).
    
    Args:
    u_rows: list of global row indices for A
    u_cols: list of global col indices for A (same as v_rows)
    v_rows: list of global row indices for B (same as u_cols)
    v_cols: list of global col indices for B
    w_rows: list of global row indices for C (same as v_cols)
    w_cols: list of global col indices for C (same as u_rows)
    
    Actually, for a block multiplication A_sub * B_sub -> C_sub:
    A_sub is (na x nb).
    B_sub is (nb x nc).
    C_sub is (na x nc).
    
    u_rows: global rows for A_sub (na items)
    u_cols: global cols for A_sub (nb items)
    v_rows: global rows for B_sub (nb items) -> should match u_cols
    v_cols: global cols for B_sub (nc items)
    w_rows: global rows for C_sub (nc items) -> should match v_cols
    w_cols: global cols for C_sub (na items) -> should match u_rows
    """
    na = len(u_rows)
    nb = len(u_cols)
    nc = len(v_cols)
    
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                u_idx = u_rows[i] * 5 + u_cols[j]
                v_idx = v_rows[j] * 5 + v_cols[k]
                w_idx = w_rows[k] * 5 + w_cols[i]
                
                U[u_idx][col] += 1
                V[v_idx][col] += 1
                W[w_idx][col] += 1
                col += 1
    return col

def _add_4x4_strassen(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col):
    """
    Add rank-56 decomposition for 4x4 block multiplication.
    Splits 4x4 into four 2x2 blocks and uses Strassen for each product term.
    """
    # A11 (top-left 2x2) * B11 (top-left 2x2) -> C11 (top-left 2x2)
    col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col)
    
    # A12 (top-right 2x2) * B21 (bottom-left 2x2) -> C11 (top-left 2x2)
    # A12 is at u_r0, u_c0+2
    # B21 is at v_r0+2, v_c0
    # C11 is at w_r0, w_c0
    col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0+2, v_r0+2, v_c0, w_r0, w_c0, col)
    
    # A11 (top-left) * B12 (top-right) -> C12 (top-right 2x2)
    # A11 at u_r0, u_c0
    # B12 at v_r0, v_c0+2
    # C12 at w_r0, w_c0+2
    col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0+2, w_r0, w_c0+2, col)
    
    # A12 (top-right) * B22 (bottom-right) -> C12 (top-right)
    # A12 at u_r0, u_c0+2
    # B22 at v_r0+2, v_c0+2
    # C12 at w_r0, w_c0+2
    col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0+2, v_r0+2, v_c0+2, w_r0, w_c0+2, col)
    
    # A21 (bottom-left) * B11 (top-left) -> C21 (bottom-left)
    # A21 at u_r0+2, u_c0
    # B11 at v_r0, v_c0
    # C21 at w_r0+2, w_c0
    col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0, v_r0, v_c0, w_r0+2, w_c0, col)
    
    # A22 (bottom-right) * B21 (bottom-left) -> C21 (bottom-left)
    # A22 at u_r0+2, u_c0+2
    # B21 at v_r0+2, v_c0
    # C21 at w_r0+2, w_c0
    col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0+2, v_r0+2, v_c0, w_r0+2, w_c0, col)
    
    # A21 (bottom-left) * B12 (top-right) -> C22 (bottom-right)
    # A21 at u_r0+2, u_c0
    # B12 at v_r0, v_c0+2
    # C22 at w_r0+2, w_c0+2
    col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0, v_r0, v_c0+2, w_r0+2, w_c0+2, col)
    
    # A22 (bottom-right) * B22 (bottom-right) -> C22 (bottom-right)
    # A22 at u_r0+2, u_c0+2
    # B22 at v_r0+2, v_c0+2
    # C22 at w_r0+2, w_c0+2
    col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0+2, v_r0+2, v_c0+2, w_r0+2, w_c0+2, col)
    
    return col

def _block_decomposition_1_4(n, m, p):
    """
    Decompose 5x5x5 using 1+4 block split.
    Rank 113.
    """
    total_rank = 1 + 4 + 16 + 4 + 16 + 16 + 56  # = 113
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Indices for 1+4 split:
    # Block 1 (size 1): indices 0
    # Block 2 (size 4): indices 1, 2, 3, 4
    
    # 1. A11(1x1) * B11(1x1) -> C11(1x1)
    # A11: row 0, col 0. Index 0.
    # B11: row 0, col 0. Index 0.
    # C11: row 0, col 0. Index 0.
    U[0][col] += 1
    V[0][col] += 1
    W[0][col] += 1
    col += 1
    
    # 2. A11(1x1) * B12(1x4) -> C12(1x4)
    # A11: row 0, col 0. Index 0.
    # B12: row 0, cols 1-4. Indices 1, 2, 3, 4.
    # C12: rows 1-4, col 0. Indices 5, 6, 7, 8. (k'*5 + i' -> k'=1..4, i'=0)
    # Schoolbook 1*1*4 = 4
    u_rows = [0]
    u_cols = [0]
    v_rows = [0]
    v_cols = [1, 2, 3, 4]
    w_rows = [1, 2, 3, 4]
    w_cols = [0]
    col = _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col)
    
    # 3. A12(1x4) * B22(4x4) -> C12(1x4)
    # A12: row 0, cols 1-4. Indices 1, 2, 3, 4.
    # B22: rows 1-4, cols 1-4.
    # C12: rows 1-4, col 0.
    # Schoolbook 1*4*4 = 16
    u_rows = [0]
    u_cols = [1, 2, 3, 4]
    v_rows = [1, 2, 3, 4]
    v_cols = [1, 2, 3, 4]
    w_rows = [1, 2, 3, 4]
    w_cols = [0]
    col = _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col)
    
    # 4. A21(4x1) * B11(1x1) -> C21(4x1)
    # A21: rows 1-4, col 0. Indices 5, 10, 15, 20.
    # B11: row 0, col 0. Index 0.
    # C21: rows 0, cols 1-4. Indices 1, 2, 3, 4. (k'=0, i'=1..4)
    # Schoolbook 4*1*1 = 4
    u_rows = [1, 2, 3, 4]
    u_cols = [0]
    v_rows = [0]
    v_cols = [0]
    w_rows = [0]
    w_cols = [1, 2, 3, 4]
    col = _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col)
    
    # 5. A22(4x4) * B21(4x1) -> C21(4x1)
    # A22: rows 1-4, cols 1-4.
    # B21: rows 1-4, col 0.
    # C21: rows 0, cols 1-4.
    # Schoolbook 4*4*1 = 16
    u_rows = [1, 2, 3, 4]
    u_cols = [1, 2, 3, 4]
    v_rows = [1, 2, 3, 4]
    v_cols = [0]
    w_rows = [0]
    w_cols = [1, 2, 3, 4]
    col = _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col)
    
    # 6. A21(4x1) * B12(1x4) -> C22(4x4)
    # A21: rows 1-4, col 0.
    # B12: row 0, cols 1-4.
    # C22: rows 1-4, cols 1-4.
    # Schoolbook 4*1*4 = 16
    u_rows = [1, 2, 3, 4]
    u_cols = [0]
    v_rows = [0]
    v_cols = [1, 2, 3, 4]
    w_rows = [1, 2, 3, 4]
    w_cols = [1, 2, 3, 4]
    col = _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col)
    
    # 7. A22(4x4) * B22(4x4) -> C22(4x4)
    # A22: rows 1-4, cols 1-4.
    # B22: rows 1-4, cols 1-4.
    # C22: rows 1-4, cols 1-4.
    # Use 4x4 Strassen (rank 56)
    col = _add_4x4_strassen(U, V, W, 1, 1, 1, 1, 1, 1, col)
    
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    
    return U, V, W

def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses 1+4 block decomposition with 4x4 Strassen, achieving rank 113.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Default: schoolbook (rank 125)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try 1+4 block decomposition (rank 113)
    try:
        U, V, W = _block_decomposition_1_4(n, m, p)
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    if time.time() < deadline:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
