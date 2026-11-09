import numpy as np
import time
import random
import sys

DIMENSIONS = [4, 4, 5]

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of (n x m) * (m x p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, trials=3):
    """Check that the decomposition is correct."""
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val += np.dot(u_row, v_row * w_row)
                C_computed[i, k] = val
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def _strassen_2x2(U, V, W, offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
    """
    Add a 2x2 Strassen decomposition to U, V, W starting at column 'offset'.
    
    a_rows, a_cols: (2, 2) array of row indices for A submatrix
    b_rows, b_cols: (2, 2) array of row indices for B submatrix  
    c_rows, c_cols: (2, 2) array of row indices for C submatrix
    
    Returns the number of columns used (7 for Strassen).
    """
    # Strassen's 7 products for 2x2 matrix multiplication:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    a00, a01 = a_rows[0, 0], a_rows[0, 1]
    a10, a11 = a_rows[1, 0], a_rows[1, 1]
    b00, b01 = b_rows[0, 0], b_rows[0, 1]
    b10, b11 = b_rows[1, 0], b_rows[1, 1]
    c00, c01 = c_rows[0, 0], c_rows[0, 1]
    c10, c11 = c_rows[1, 0], c_rows[1, 1]
    
    col = offset
    # P1 = (A00 + A11) * (B00 + B11)
    U[a00, col] += 1.0
    U[a11, col] += 1.0
    V[b00, col] += 1.0
    V[b11, col] += 1.0
    W[c00, col] += 1.0
    W[c11, col] += 1.0
    col += 1
    
    # P2 = (A10 + A11) * B00
    U[a10, col] += 1.0
    U[a11, col] += 1.0
    V[b00, col] += 1.0
    W[c10, col] += 1.0
    col += 1
    
    # P3 = A00 * (B01 - B11)
    U[a00, col] += 1.0
    V[b01, col] += 1.0
    V[b11, col] -= 1.0
    W[c01, col] += 1.0
    col += 1
    
    # P4 = A11 * (B10 - B00)
    U[a11, col] += 1.0
    V[b10, col] += 1.0
    V[b00, col] -= 1.0
    W[c00, col] += 1.0
    W[c10, col] += 1.0
    col += 1
    
    # P5 = (A00 + A01) * B11
    U[a00, col] += 1.0
    U[a01, col] += 1.0
    V[b11, col] += 1.0
    W[c00, col] -= 1.0
    W[c01, col] += 1.0
    col += 1
    
    # P6 = (A10 - A00) * (B00 + B01)
    U[a10, col] += 1.0
    U[a00, col] -= 1.0
    V[b00, col] += 1.0
    V[b01, col] += 1.0
    W[c11, col] += 1.0
    col += 1
    
    # P7 = (A01 - A11) * (B10 + B11)
    U[a01, col] += 1.0
    U[a11, col] -= 1.0
    V[b10, col] += 1.0
    V[b11, col] += 1.0
    W[c00, col] += 1.0
    col += 1
    
    return col - offset  # 7 columns used

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s kill
    
    n, m, p = DIMENSIONS
    N = n * m  # 16
    M = m * p  # 20
    P = p * n  # 20
    
    # Start with schoolbook decomposition (rank 80)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: Use Strassen for the 4x4x4 sub-problem
    # Split the 4x4x5 problem into:
    # - 4x4x4 part (columns 0-3 of p)
    # - 4x4x1 part (column 4 of p)
    
    # For the 4x4x4 part, use Strassen at the top level:
    # Split 4x4 into 2x2 blocks:
    # A = [A00 A01; A10 A11] where Aij is 2x2
    # B = [B00 B01; B10 B11] where Bij is 2x2
    
    # For 4x4x4:
    # A rows: 0-3, B cols: 0-3
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # For the 4x4x1 part:
    # B has only col 4, so B00: rows 0-1, col 4; B10: rows 2-3, col 4
    # This is 4x4x1, which we can do with schoolbook (rank 16)
    
    # Let's try a hybrid approach:
    # Use Strassen for 4x4x4 (rank 7 per 2x2 block, but we need to be careful)
    
    # Actually, let's try a different structural approach:
    # Use the fact that we can decompose 4x4x5 as a combination of smaller problems
    
    # For now, let's try to find a valid decomposition with rank < 80
    # by using a randomized approach with a different structure
    
    # Try using block decomposition:
    # Split A into left (cols 0-1) and right (cols 2-3)
    # Split B into top (rows 0-1) and bottom (rows 2-3)
    # Split C into four 2x2 quadrants
    
    # This gives us:
    # C00 = A_left * B_top  (2x2 * 2x2)
    # C01 = A_right * B_top  (2x2 * 2x2)
    # C10 = A_left * B_bottom  (2x2 * 2x2)
    # C11 = A_right * B_bottom  (2x2 * 2x2)
    
    # For 4x4x5, B has 5 columns, so:
    # B_top: rows 0-1, cols 0-4 (2x5)
    # B_bottom: rows 2-3, cols 0-4 (2x5)
    
    # C00: rows 0-1, cols 0-4 (2x5)
    # C01: rows 0-1, cols 0-4 (2x5)
    # C10: rows 2-3, cols 0-4 (2x5)
    # C11: rows 2-3, cols 0-4 (2x5)
    
    # Each 2x2 * 2x5 product has rank 10 (schoolbook) or can be done more efficiently
    # 2x2 * 2x5: A is 2x2, B is 2x5, C is 2x5
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is 2 mults per output entry, 2*5=10 outputs, so rank 10
    
    # So 4 blocks * rank 10 = rank 40 for the 4x4x5 problem!
    # This is a huge improvement over 80!
    
    # Let's implement this:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:5] -> C[0:2, 0:5]
    # Block 2: A[0:2, 2:4] * B[0:2, 0:5] -> C[0:2, 0:5] (but this adds to same C)
    # Wait, this doesn't work because C gets contributions from multiple blocks
    
    # Actually, the correct decomposition is:
    # C = A * B
    # A is 4x4, B is 4x5
    # Split A into A_left (4x2) and A_right (4x2)
    # Split B into B_top (2x5) and B_bottom (2x5)
    
    # C = A_left * B_top + A_right * B_bottom
    # No, that's not right either.
    
    # Let me think more carefully:
    # A = [A00 A01] where A00, A01 are 4x2
    # B = [B00; B01] where B00, B01 are 2x5
    # C = A00 * B00 + A01 * B01
    # A00 * B00 is 4x5, A01 * B01 is 4x5
    
    # A00 is 4x2, B00 is 2x5: rank = 4*2*5/2 = 20 (schoolbook)
    # But we can do better: 4x2 * 2x5 can be done with rank 10
    # Because C[i,k] = sum_j A00[i,j] * B00[j,k], j=0,1
    # So C[i,k] = A00[i,0]*B00[0,k] + A00[i,1]*B00[1,k]
    # This is 2 mults per (i,k), 4*5=20 outputs, so rank 20
    
    # Similarly for A01 * B01: rank 20
    # Total rank = 40
    
    # Let's implement this:
    # A00: rows 0-3, cols 0-1 of A -> U indices 0-7 (i*m + j, i=0-3, j=0-1)
    # B00: rows 0-1, cols 0-4 of B -> V indices 0-9 (j*p + k, j=0-1, k=0-4)
    # C00: rows 0-3, cols 0-4 of C -> W indices (k*n + i, k=0-4, i=0-3)
    
    # For A00 * B00:
    # U: rows 0-7 (i=0-3, j=0-1)
    # V: rows 0-9 (j=0-1, k=0-4)
    # W: rows (k*n + i) for k=0-4, i=0-3
    
    col = 0
    # A00 * B00 part
    for i in range(4):
        for j in range(2):  # j = 0, 1
            for k in range(5):  # k = 0-4
                U[i * 4 + j, col] = 1.0  # A00[i,j]
                V[j * 5 + k, col] = 1.0  # B00[j,k]
                W[k * 4 + i, col] = 1.0  # C[i,k]
                col += 1
    
    # A01 * B01 part
    for i in range(4):
        for j in range(2):  # j = 0, 1 (but now j corresponds to cols 2-3 of A)
            for k in range(5):  # k = 0-4
                U[i * 4 + (j + 2), col] = 1.0  # A01[i,j] = A[i, j+2]
                V[(j + 2) * 5 + k, col] = 1.0  # B01[j,k] = B[j+2, k]
                W[k * 4 + i, col] = 1.0  # C[i,k]
                col += 1
    
    rank = col
    # rank should be 40
    
    if rank < best_rank:
        if _verify(U, V, W, n, m, p, trials=3):
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try to improve further using Strassen for the 4x2 * 2x5 sub-problems
    # 4x2 * 2x5: Can we do better than rank 20?
    # The tensor is M_{4,2,5}, which has rank at most 4*2*5 = 40, but we're using 20
    # Actually, for 4x2 * 2x5, the rank is exactly 20 (schoolbook is optimal for this)
    # Because it's equivalent to 20 independent scalar mults
    
    # So rank 40 is likely close to optimal for this approach
    
    # Let's try another approach: use Strassen for 4x4x4 and add 4x4x1
    # 4x4x4 with Strassen at top level:
    # Split into 2x2 blocks, use Strassen (7 products)
    # Each 2x2 * 2x2 product can be done with rank 7 (Strassen) or 8 (schoolbook)
    # If we use Strassen for the 2x2 blocks: 7 * 7 = 49
    # If we use schoolbook for the 2x2 blocks: 7 * 8 = 56
    
    # For 4x4x1: rank 16 (schoolbook)
    # Total: 49 + 16 = 65 or 56 + 16 = 72
    
    # 65 < 40? No, 65 > 40. So the block approach (rank 40) is better!
    
    # Let's verify the rank 40 decomposition is correct
    if not _verify(best_U, best_V, best_W, n, m, p, trials=5):
        # Fallback to schoolbook if something went wrong
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try to find even lower rank using random search
    # But given time constraints, return what we have
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
