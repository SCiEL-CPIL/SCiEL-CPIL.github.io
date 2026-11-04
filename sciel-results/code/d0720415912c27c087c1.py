"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 120

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where U is (4,7), V is (4,7), W is (4,7).
    U[i*2+j] is the coefficient for A[i,j], V[i*2+j] for B[i,j], W[i*2+j] for C[i,j].
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]  # A factors: A00,A01,A10,A11
    V = [[0]*7 for _ in range(4)]  # B factors: B00,B01,B10,B11
    W = [[0]*7 for _ in range(4)]  # C factors: C00,C01,C10,C11
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for nxm x mxp -> nxp.
    
    Returns (U, V, W) where U is (n*m, n*m*p), V is (m*p, n*m*p), W is (p*n, n*m*p).
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions with matching dimensions.
    
    Each decomp has U of shape (rows, r_i), V of shape (mcols, r_i), W of shape (prows, r_i).
    Returns combined (U, V, W) with total rank = sum(r_i).
    """
    if not decomps:
        return None, None, None
    n_rows = decomps[0][0].__len__()
    n_mcols = decomps[0][1].__len__()
    n_prows = decomps[0][2].__len__()
    total_rank = sum(d[0].__len__() if False else len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n_rows)]
    V = [[0]*total_rank for _ in range(n_mcols)]
    W = [[0]*total_rank for _ in range(n_prows)]
    
    offset = 0
    for dU, dV, dW in decomps:
        r = len(dU[0])
        for i in range(n_rows):
            for c in range(r):
                U[i][offset+c] = dU[i][c]
        for i in range(n_mcols):
            for c in range(r):
                V[i][offset+c] = dV[i][c]
        for i in range(n_prows):
            for c in range(r):
                W[i][offset+c] = dW[i][c]
        offset += r
    
    return U, V, W

def _build_rank74():
    """Build rank-74 decomposition using Strassen on 2x2x2 sub-blocks.
    
    Split A (3x5) into rows [A1 (2x5), A2 (1x5)]
    Split B (5x5) into cols [B1 (5x2), B2 (5x3)]
    Split inner dim 5 = 2 + 3
    
    C11 (2x2) = A1*B1: inner 2+3 -> 2x2x2 (Strassen 7) + 2x3x2 (schoolbook 12) = 19
    C12 (2x3) = A1*B2: inner 2+3 -> 2x2x3 (schoolbook 12) + 2x3x3 (schoolbook 18) = 30
    C21 (1x2) = A2*B1: inner 2+3 -> 1x2x2 (schoolbook 4) + 1x3x2 (schoolbook 6) = 10
    C22 (1x3) = A2*B2: inner 2+3 -> 1x2x3 (schoolbook 6) + 1x3x3 (schoolbook 9) = 15
    Total: 74
    """
    # We need to build the full 3x5x5 decomposition.
    # U has shape (15, rank), V has shape (25, rank), W has shape (15, rank).
    # U index: i*5+j for A[i,j], i in 0..2, j in 0..4
    # V index: j*5+k for B[j,k], j in 0..4, k in 0..4
    # W index: k*3+i for C[i,k], i in 0..2, k in 0..4
    
    all_decomps = []
    
    # For each sub-problem, we need to map local indices to global indices.
    # Let's build each block's contribution directly.
    
    def add_block(a_row_start, a_rows, b_col_start, b_cols, inner_start, inner_len, 
                  c_row_start, c_rows, c_col_start, c_cols, use_strassen=False):
        """Add a block decomposition.
        
        a_row_start, a_rows: rows of A used (from a_row_start, a_rows rows)
        b_col_start, b_cols: cols of B used (from b_col_start, b_cols cols)
        inner_start, inner_len: inner dimension slice
        c_row_start, c_rows: rows of C
        c_col_start, c_cols: cols of C
        """
        n_sub = a_rows
        m_sub = inner_len
        p_sub = b_cols
        c_n = c_rows
        c_p = c_cols
        
        if use_strassen and n_sub == 2 and m_sub == 2 and p_sub == 2:
            sU, sV, sW = _strassen_222()
            # Map local indices to global
            for local_i in range(4):
                gi = local_i // 2  # local row
                gj = local_i % 2   # local col (inner)
                global_a_row = a_row_start + gi
                global_a_inner = inner_start + gj
                for c in range(7):
                    all_decomps.append((global_a_row * 5 + global_a_inner, sU[local_i][c]))
            
            for local_j in range(4):
                gj = local_j // 2  # local row (inner)
                gk = local_j % 2   # local col
                global_b_inner = inner_start + gj
                global_b_col = b_col_start + gk
                for c in range(7):
                    all_decomps.append((global_b_inner * 5 + global_b_col, sV[local_j][c]))
            
            for local_k in range(4):
                gk = local_k // 2  # local col
                gi = local_k % 2   # local row
                global_c_col = c_col_start + gk
                global_c_row = c_row_start + gi
                for c in range(7):
                    all_decomps.append((global_c_col * 3 + global_c_row, sW[local_k][c]))
        else:
            # Schoolbook
            for gi in range(c_n):
                for gj in range(m_sub):
                    for gk in range(c_p):
                        global_a_row = a_row_start + gi
                        global_a_inner = inner_start + gj
                        global_b_inner = inner_start + gj
                        global_b_col = b_col_start + gk
                        global_c_col = c_col_start + gk
                        global_c_row = c_row_start + gi
                        all_decomps.append((global_a_row * 5 + global_a_inner, 1))
                        all_decomps.append((global_b_inner * 5 + global_b_col, 1))
                        all_decomps.append((global_c_col * 3 + global_c_row, 1))
    
    # C11 (2x2) = A1*B1: A rows 0-1, B cols 0-1, inner 0-4
    # Inner split: 0-1 (len 2) and 2-4 (len 3)
    # Part 1: inner 0-1, use Strassen (2x2x2)
    add_block(a_row_start=0, a_rows=2, b_col_start=0, b_cols=2, 
              inner_start=0, inner_len=2, c_row_start=0, c_rows=2, c_col_start=0, c_cols=2, 
              use_strassen=True)
    # Part 2: inner 2-4, schoolbook (2x3x2)
    add_block(a_row_start=0, a_rows=2, b_col_start=0, b_cols=2, 
              inner_start=2, inner_len=3, c_row_start=0, c_rows=2, c_col_start=0, c_cols=2, 
              use_strassen=False)
    
    # C12 (2x3) = A1*B2: A rows 0-1, B cols 2-4, inner 0-4
    # Inner split: 0-1 (len 2) and 2-4 (len 3)
    add_block(a_row_start=0, a_rows=2, b_col_start=2, b_cols=3, 
              inner_start=0, inner_len=2, c_row_start=0, c_rows=2, c_col_start=2, c_cols=3, 
              use_strassen=False)
    add_block(a_row_start=0, a_rows=2, b_col_start=2, b_cols=3, 
              inner_start=2, inner_len=3, c_row_start=0, c_rows=2, c_col_start=2, c_cols=3, 
              use_strassen=False)
    
    # C21 (1x2) = A2*B1: A row 2, B cols 0-1, inner 0-4
    add_block(a_row_start=2, a_rows=1, b_col_start=0, b_cols=2, 
              inner_start=0, inner_len=2, c_row_start=2, c_rows=1, c_col_start=0, c_cols=2, 
              use_strassen=False)
    add_block(a_row_start=2, a_rows=1, b_col_start=0, b_cols=2, 
              inner_start=2, inner_len=3, c_row_start=2, c_rows=1, c_col_start=0, c_cols=2, 
              use_strassen=False)
    
    # C22 (1x3) = A2*B2: A row 2, B cols 2-4, inner 0-4
    add_block(a_row_start=2, a_rows=1, b_col_start=2, b_cols=3, 
              inner_start=0, inner_len=2, c_row_start=2, c_rows=1, c_col_start=2, c_cols=3, 
              use_strassen=False)
    add_block(a_row_start=2, a_rows=1, b_col_start=2, b_cols=3, 
              inner_start=2, inner_len=3, c_row_start=2, c_rows=1, c_col_start=2, c_cols=3, 
              use_strassen=False)
    
    # Now build U, V, W from all_decomps
    # all_decomps is a flat list of (index, value) triples grouped in threes.
    # Let's restructure: each "term" contributes to U, V, W.
    
    # Rebuild: collect terms properly
    terms = []  # list of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    def rebuild():
        terms.clear()
        # Re-do the block additions collecting terms properly
        pass
    
    # Actually, let me just directly build U, V, W arrays.
    # Total rank = 19 + 30 + 10 + 15 = 74
    total_rank = 74
    U = [[0]*total_rank for _ in range(15)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(15)]
    
    col = 0
    
    # C11 part 1: Strassen 2x2x2 (7 terms)
    sU, sV, sW = _strassen_222()
    for local_i in range(4):
        gi = local_i // 2
        gj = local_i % 2
        a_row = 0 + gi
        a_inner = 0 + gj
        for c in range(7):
            U[a_row * 5 + a_inner][col + c] = sU[local_i][c]
    for local_j in range(4):
        gj = local_j // 2
        gk = local_j % 2
        b_inner = 0 + gj
        b_col = 0 + gk
        for c in range(7):
            V[b_inner * 5 + b_col][col + c] = sV[local_j][c]
    for local_k in range(4):
        gk = local_k // 2
        gi = local_k % 2
        c_col = 0 + gk
        c_row = 0 + gi
        for c in range(7):
            W[c_col * 3 + c_row][col + c] = sW[local_k][c]
    col += 7
    
    # C11 part 2: schoolbook 2x3x2 (12 terms)
    for gi in range(2):
        for gj in range(3):
            for gk in range(2):
                a_idx = (0+gi)*5 + (2+gj)
                b_idx = (2+gj)*5 + (0+gk)
                w_idx = (0+gk)*3 + (0+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C12 part 1: schoolbook 2x2x3 (12 terms)
    for gi in range(2):
        for gj in range(2):
            for gk in range(3):
                a_idx = (0+gi)*5 + (0+gj)
                b_idx = (0+gj)*5 + (2+gk)
                w_idx = (2+gk)*3 + (0+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C12 part 2: schoolbook 2x3x3 (18 terms)
    for gi in range(2):
        for gj in range(3):
            for gk in range(3):
                a_idx = (0+gi)*5 + (2+gj)
                b_idx = (2+gj)*5 + (2+gk)
                w_idx = (2+gk)*3 + (0+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C21 part 1: schoolbook 1x2x2 (4 terms)
    for gi in range(1):
        for gj in range(2):
            for gk in range(2):
                a_idx = (2+gi)*5 + (0+gj)
                b_idx = (0+gj)*5 + (0+gk)
                w_idx = (0+gk)*3 + (2+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C21 part 2: schoolbook 1x3x2 (6 terms)
    for gi in range(1):
        for gj in range(3):
            for gk in range(2):
                a_idx = (2+gi)*5 + (2+gj)
                b_idx = (2+gj)*5 + (0+gk)
                w_idx = (0+gk)*3 + (2+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C22 part 1: schoolbook 1x2x3 (6 terms)
    for gi in range(1):
        for gj in range(2):
            for gk in range(3):
                a_idx = (2+gi)*5 + (0+gj)
                b_idx = (0+gj)*5 + (2+gk)
                w_idx = (2+gk)*3 + (2+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # C22 part 2: schoolbook 1x3x3 (9 terms)
    for gi in range(1):
        for gj in range(3):
            for gk in range(3):
                a_idx = (2+gi)*5 + (2+gj)
                b_idx = (2+gj)*5 + (2+gk)
                w_idx = (2+gk)*3 + (2+gi)
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result):
    """Verify that the decomposition is correct."""
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    n, m, p = 3, 5, 5
    
    # Reconstruct the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with identity tensor: T[i,j,k] = 1 if j==k else 0... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # So T should be such that sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # T[i,j,k] = 1 for all i,j,k
    
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all i,j,k
    # But wait, W is indexed as W[k*n+i] per the trace convention.
    
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    errors += 1
                    max_err = max(max_err, abs(val - 1))
    
    if errors > 0:
        print(f"Verification FAILED: {errors} errors, max error {max_err}")
        return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication."""
    global _DEADLINE
    
    try:
        result = _build_rank74()
        
        if time.time() > _DEADLINE - 10:
            return result
        
        # Verify
        if not _verify_decomposition(result):
            # Fall back to schoolbook
            n, m, p = DIMENSIONS
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
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
