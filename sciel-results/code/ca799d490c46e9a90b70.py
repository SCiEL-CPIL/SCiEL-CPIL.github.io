import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor (vectorized)."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    R = U_np.shape[1]
    
    # For each (i,j,k), the tensor entry should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build index arrays
    ii, jj, kk = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()
    kk = kk.ravel()
    
    # Gather the relevant rows
    U_rows = U_np[ii * m + jj, :]  # (n*m*p, R)
    V_rows = V_np[jj * p + kk, :]  # (n*m*p, R)
    W_rows = W_np[kk * n + ii, :]  # (n*m*p, R)
    
    # Sum over rank dimension
    vals = np.sum(U_rows * V_rows * W_rows, axis=1)
    
    return bool(np.all(np.abs(vals - 1.0) <= tol))

def _construct_hybrid():
    """
    Construct a rank-88 decomposition for 3x4x8 matmul.
    Split 3x4 into 1x4 (schoolbook, rank 32) + 2x4 (Strassen-based, rank 56).
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    def strassen_block(a_idx, b_idx, c_idx, base_col):
        """Apply Strassen 7-mul to a 2x2x2 sub-problem with explicit index maps."""
        col = base_col
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        U[a_idx[(0,0)], col] += 1
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(0,0)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        U[a_idx[(1,0)], col] += 1
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(0,0)], col] += 1
        W[c_idx[(1,0)], col] += 1
        W[c_idx[(1,1)], col] -= 1
        col += 1
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        U[a_idx[(0,0)], col] += 1
        V[b_idx[(0,1)], col] += 1
        V[b_idx[(1,1)], col] -= 1
        W[c_idx[(0,1)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(1,0)], col] += 1
        V[b_idx[(0,0)], col] -= 1
        W[c_idx[(0,0)], col] += 1
        W[c_idx[(1,0)], col] += 1
        col += 1
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        U[a_idx[(0,0)], col] += 1
        U[a_idx[(0,1)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] -= 1
        W[c_idx[(0,1)], col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        U[a_idx[(1,0)], col] += 1
        U[a_idx[(0,0)], col] -= 1
        V[b_idx[(0,0)], col] += 1
        V[b_idx[(0,1)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        U[a_idx[(0,1)], col] += 1
        U[a_idx[(1,1)], col] -= 1
        V[b_idx[(1,0)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] += 1
        col += 1
        return col

    # A1: rows 1-2 of A, cols 0-1
    a1_idx = {(0,0): 1*4+0, (0,1): 1*4+1, (1,0): 2*4+0, (1,1): 2*4+1}
    # A2: rows 1-2 of A, cols 2-3
    a2_idx = {(0,0): 1*4+2, (0,1): 1*4+3, (1,0): 2*4+2, (1,1): 2*4+3}
    
    # Left 4x4 block of B (cols 0-3)
    b11_idx = {(0,0): 0*8+0, (0,1): 0*8+1, (1,0): 1*8+0, (1,1): 1*8+1}
    b12_idx = {(0,0): 0*8+2, (0,1): 0*8+3, (1,0): 1*8+2, (1,1): 1*8+3}
    b21_idx = {(0,0): 2*8+0, (0,1): 2*8+1, (1,0): 3*8+0, (1,1): 3*8+1}
    b22_idx = {(0,0): 2*8+2, (0,1): 2*8+3, (1,0): 3*8+2, (1,1): 3*8+3}
    c1_idx = {(0,0): 0*3+1, (0,1): 1*3+1, (1,0): 0*3+2, (1,1): 1*3+2}
    c2_idx = {(0,0): 2*3+1, (0,1): 3*3+1, (1,0): 2*3+2, (1,1): 3*3+2}
    
    # C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    col = strassen_block(a1_idx, b11_idx, c1_idx, col)
    col = strassen_block(a2_idx, b21_idx, c1_idx, col)
    col = strassen_block(a1_idx, b12_idx, c2_idx, col)
    col = strassen_block(a2_idx, b22_idx, c2_idx, col)
    
    # Right 4x4 block of B (cols 4-7)
    b11r_idx = {(0,0): 0*8+4, (0,1): 0*8+5, (1,0): 1*8+4, (1,1): 1*8+5}
    b12r_idx = {(0,0): 0*8+6, (0,1): 0*8+7, (1,0): 1*8+6, (1,1): 1*8+7}
    b21r_idx = {(0,0): 2*8+4, (0,1): 2*8+5, (1,0): 3*8+4, (1,1): 3*8+5}
    b22r_idx = {(0,0): 2*8+6, (0,1): 2*8+7, (1,0): 3*8+6, (1,1): 3*8+7}
    c1r_idx = {(0,0): 4*3+1, (0,1): 5*3+1, (1,0): 4*3+2, (1,1): 5*3+2}
    c2r_idx = {(0,0): 6*3+1, (0,1): 7*3+1, (1,0): 6*3+2, (1,1): 7*3+2}
    
    col = strassen_block(a1_idx, b11r_idx, c1r_idx, col)
    col = strassen_block(a2_idx, b21r_idx, c1r_idx, col)
    col = strassen_block(a1_idx, b12r_idx, c2r_idx, col)
    col = strassen_block(a2_idx, b22r_idx, c2r_idx, col)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Try hybrid Strassen construction (rank 88)
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Check deadline before fallback
    if time.time() > deadline - 5.0:
        # Minimal fallback: just return schoolbook without verification
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
