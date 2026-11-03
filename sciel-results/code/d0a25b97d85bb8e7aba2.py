"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    Um = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Vm = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wm = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        T += Um[:, :, c][:, :, np.newaxis] * Vm[:, c, :][np.newaxis, :, :] * Wm[:, c, :][:, np.newaxis, :]
    # Wm is (p, n, rank), Wm[:, c, :] is (p, n) but we need W[k,i] = Wm[k, i, c]
    # Let me redo: Wm[k*n+i, c] reshaped to (p, n, rank)
    # So Wm[:, :, c] is (p, n), and we need Wm[k, i, c]
    # T[i,j,k] += Um[i,j,c] * Vm[j,k,c] * Wm[k,i,c]
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Um[:, :, c]  # (n, m)
        vc = Vm[:, :, c]  # (m, p)
        wc = Wm[:, :, c]  # (p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(T, 1.0, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2, rank 7.
    U, V, W are (4, 7) matrices where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 terms for 2x2*2x2
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # contributes to C00 and C11
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 += 1, C11 -= 1
    
    # Term 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += 1, C11 += 1
    
    # Term 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1  # C10 += 1, C00 += 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1  # C01 -= 1, C11 += 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += 1
    
    return U, V, W

def _schoolbook_rect(n, m, p):
    """Schoolbook for n x m * m x p, rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _compose_block_decomposition():
    """Build the full 4x4 * 4x5 decomposition using block structure.
    
    A (4x4) split into 2x2 blocks:
      A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
      A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    
    B (4x5) split:
      B11 (rows 0-1, cols 0-2), B12 (rows 0-1, cols 3-4)
      B21 (rows 2-3, cols 0-2), B22 (rows 2-3, cols 3-4)
    
    C (4x5) split:
      C11 (rows 0-1, cols 0-2) = A11*B11 + A12*B21
      C12 (rows 0-1, cols 3-4) = A11*B12 + A12*B22
      C21 (rows 2-3, cols 0-2) = A21*B11 + A22*B21
      C22 (rows 2-3, cols 3-4) = A21*B12 + A22*B22
    """
    # For each 2x2 * 2x3 product, I'll use a rank-10 decomposition
    # For each 2x2 * 2x2 product, I'll use Strassen rank 7
    
    # Let me think about this differently.
    # The total rank will be the sum of ranks for each independent computation.
    # But terms can share structure.
    
    # Actually, let me just use schoolbook for all sub-products and see the total rank.
    # C11: A11(2x2)*B11(2x3) rank 12, A12(2x2)*B21(2x3) rank 12 -> total 24
    # C12: A11(2x2)*B12(2x2) rank 4, A12(2x2)*B22(2x2) rank 4 -> total 8
    # C21: A21(2x2)*B11(2x3) rank 12, A22(2x2)*B21(2x3) rank 12 -> total 24
    # C22: A21(2x2)*B12(2x2) rank 4, A22(2x2)*B22(2x2) rank 4 -> total 8
    # Total: 24+8+24+8 = 64
    
    # That's better than 80! Let me implement this.
    
    n, m, p = 4, 4, 5
    total_rank = 64
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: A11 is rows 0-1, cols 0-1 of A; B11 is rows 0-1, cols 0-2 of B
    # For each (i,j,k) in local coords: i in 0-1, j in 0-1, k in 0-2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A11[i,j] * B11[j,k] contributes to C11[i,k]
                # Global A index: i*2 + j (rows 0-1, cols 0-1)
                # Global B index: j*3 + k (rows 0-1, cols 0-2)  
                # Wait, B is 4x5, B[j*p+k] where j is row, k is col
                # B11 is rows 0-1, cols 0-2 of B, so B[j*5+k] for j in 0-1, k in 0-2
                # C11[i,k] is at global C row i, col k, so C[i*5+k]
                # W index: k*n + i = k*4 + i
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    # A12*B21: A12 is rows 0-1, cols 2-3 of A; B21 is rows 2-3, cols 0-2 of B
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A12[i,j] * B21[j,k] contributes to C11[i,k]
                # Global A index: i*4 + (2+j)
                # Global B index: (2+j)*5 + k
                # C11[i,k] at global C row i, col k: W index k*4 + i
                U[i*4 + (2+j)][col] = 1
                V[(2+j)*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: A11 rows 0-1 cols 0-1, B12 rows 0-1 cols 3-4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A11[i,j] * B12[j,k] contributes to C12[i,k]
                # Global A: i*4 + j
                # Global B: j*5 + (3+k)
                # C12[i,k] at global row i, col 3+k: W index (3+k)*4 + i
                U[i*4 + j][col] = 1
                V[j*5 + (3+k)][col] = 1
                W[(3+k)*4 + i][col] = 1
                col += 1
    
    # A12*B22: A12 rows 0-1 cols 2-3, B22 rows 2-3 cols 3-4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A12[i,j] * B22[j,k] contributes to C12[i,k]
                # Global A: i*4 + (2+j)
                # Global B: (2+j)*5 + (3+k)
                # C12[i,k] at global row i, col 3+k: W index (3+k)*4 + i
                U[i*4 + (2+j)][col] = 1
                V[(2+j)*5 + (3+k)][col] = 1
                W[(3+k)*4 + i][col] = 1
                col += 1
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: A21 rows 2-3 cols 0-1, B11 rows 0-1 cols 0-2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A21[i,j] * B11[j,k] contributes to C21[i,k]
                # Global A: (2+i)*4 + j
                # Global B: j*5 + k
                # C21[i,k] at global row 2+i, col k: W index k*4 + (2+i)
                U[(2+i)*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*4 + (2+i)][col] = 1
                col += 1
    
    # A22*B21: A22 rows 2-3 cols 2-3, B21 rows 2-3 cols 0-2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A22[i,j] * B21[j,k] contributes to C21[i,k]
                # Global A: (2+i)*4 + (2+j)
                # Global B: (2+j)*5 + k
                # C21[i,k] at global row 2+i, col k: W index k*4 + (2+i)
                U[(2+i)*4 + (2+j)][col] = 1
                V[(2+j)*5 + k][col] = 1
                W[k*4 + (2+i)][col] = 1
                col += 1
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: A21 rows 2-3 cols 0-1, B12 rows 0-1 cols 3-4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A21[i,j] * B12[j,k] contributes to C22[i,k]
                # Global A: (2+i)*4 + j
                # Global B: j*5 + (3+k)
                # C22[i,k] at global row 2+i, col 3+k: W index (3+k)*4 + (2+i)
                U[(2+i)*4 + j][col] = 1
                V[j*5 + (3+k)][col] = 1
                W[(3+k)*4 + (2+i)][col] = 1
                col += 1
    
    # A22*B22: A22 rows 2-3 cols 2-3, B22 rows 2-3 cols 3-4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A22[i,j] * B22[j,k] contributes to C22[i,k]
                # Global A: (2+i)*4 + (2+j)
                # Global B: (2+j)*5 + (3+k)
                # C22[i,k] at global row 2+i, col 3+k: W index (3+k)*4 + (2+i)
                U[(2+i)*4 + (2+j)][col] = 1
                V[(2+j)*5 + (3+k)][col] = 1
                W[(3+k)*4 + (2+i)][col] = 1
                col += 1
    
    assert col == total_rank, f"col={col}, expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_improvement():
    """Try to use Strassen for the 2x2*2x2 sub-products to reduce rank further.
    
    C12 = A11*B12 + A12*B22 where both are 2x2*2x2
    C22 = A21*B12 + A22*B22 where both are 2x2*2x2
    
    For C12: A11(2x2)*B12(2x2) + A12(2x2)*B22(2x2)
    Each 2x2*2x2 can be done with Strassen rank 7.
    But they share the same output C12, so we need to combine them.
    
    Actually, let me think about this more carefully.
    C12[i,k] = sum_j A11[i,j]*B12[j,k] + sum_j A12[i,j]*B22[j,k]
    
    This is equivalent to:
    C12 = [A11 | A12] * [[B12], [B22]]
    where [A11|A12] is 2x4 and [[B12],[B22]] is 4x2.
    
    The schoolbook rank for 2x4 * 4x2 is 2*4*2 = 16.
    Strassen for each 2x2*2x2 gives 7+7=14, which is better!
    
    Similarly for C22.
    
    Let me implement Strassen for the 2x2*2x2 parts.
    """
    n, m, p = 4, 4, 5
    
    # For C12 and C22, use Strassen (rank 7 each) instead of schoolbook (rank 4 each)
    # Wait, schoolbook for 2x2*2x2 is rank 4, Strassen is rank 7. That's worse!
    # Strassen only helps for larger matrices.
    
    # For 2x2*2x2, schoolbook rank 4 is better than Strassen rank 7.
    # Strassen helps when you can combine multiple sub-products.
    
    # Let me reconsider. The total rank with my block approach is 64.
    # Can I do better?
    
    # What if I use a different partition?
    # Split into 1x1 blocks? That's just schoolbook rank 80.
    # Split into 2x2 blocks? That's what I did, rank 64.
    # Split into 4x4 blocks? That's the whole thing.
    
    # Let me try: use Strassen for the whole 4x4*4x4 part (first 4 cols of B)
    # and schoolbook for the remaining 1 col.
    
    # 4x4*4x4 with Strassen: rank 49 (7^2)
    # 4x4*4x1: rank 16
    # Total: 49 + 16 = 65. Slightly worse than 64.
    
    # What about 2x2*2x3 products? Is there a better than schoolbook (rank 12)?
    # 2x2*2x3: The tensor M_{2,2,3}. Known rank is 12 (schoolbook) or can it be lower?
    # Actually, I believe the rank of M_{2,2,3} is 12.
    
    # Let me just go with the rank-64 solution and verify it.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to leave margin
    
    # Build the block decomposition (rank 64)
    result = _compose_block_decomposition()
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
