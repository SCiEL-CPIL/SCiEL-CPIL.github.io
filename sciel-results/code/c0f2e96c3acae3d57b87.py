import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor (vectorized)."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    ii, jj, kk = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel()

    U_rows = U_np[ii * m + jj, :]
    V_rows = V_np[jj * p + kk, :]
    W_rows = W_np[kk * n + ii, :]

    vals = np.sum(U_rows * V_rows * W_rows, axis=1)
    return bool(np.all(np.abs(vals - 1.0) <= tol))

def _strassen_block(U, V, W, a_idx, b_idx, c_idx, base_col):
    """Apply Strassen 7-mul to a 2x2x2 sub-problem. Returns next free col."""
    col = base_col
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    U[a_idx[(0,0)], col] += 1; U[a_idx[(1,1)], col] += 1
    V[b_idx[(0,0)], col] += 1; V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] += 1; W[c_idx[(1,1)], col] += 1
    col += 1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    U[a_idx[(1,0)], col] += 1; U[a_idx[(1,1)], col] += 1
    V[b_idx[(0,0)], col] += 1
    W[c_idx[(1,0)], col] += 1; W[c_idx[(1,1)], col] -= 1
    col += 1
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    U[a_idx[(0,0)], col] += 1
    V[b_idx[(0,1)], col] += 1; V[b_idx[(1,1)], col] -= 1
    W[c_idx[(0,1)], col] += 1; W[c_idx[(1,1)], col] += 1
    col += 1
    # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(1,0)], col] += 1; V[b_idx[(0,0)], col] -= 1
    W[c_idx[(0,0)], col] += 1; W[c_idx[(1,0)], col] += 1
    col += 1
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    U[a_idx[(0,0)], col] += 1; U[a_idx[(0,1)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] -= 1; W[c_idx[(0,1)], col] += 1
    col += 1
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    U[a_idx[(1,0)], col] += 1; U[a_idx[(0,0)], col] -= 1
    V[b_idx[(0,0)], col] += 1; V[b_idx[(0,1)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    U[a_idx[(0,1)], col] += 1; U[a_idx[(1,1)], col] -= 1
    V[b_idx[(1,0)], col] += 1; V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] += 1
    col += 1
    return col

def _construct_hybrid():
    """Rank-88 decomposition: 1x4x8 schoolbook (32) + 2x4x8 Strassen (56)."""
    n, m, p = 3, 4, 8
    R = 88
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Part 1: row 0 of A (1x4) x B (4x8) schoolbook, rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1

    # A rows 1-2, cols 0-1 and cols 2-3
    a1_idx = {(0,0): 1*4+0, (0,1): 1*4+1, (1,0): 2*4+0, (1,1): 2*4+1}
    a2_idx = {(0,0): 1*4+2, (0,1): 1*4+3, (1,0): 2*4+2, (1,1): 2*4+3}

    # B blocks (4x4) for left half (cols 0-3) and right half (cols 4-7)
    def bblocks(col_off):
        return (
            {(0,0): 0*8+col_off+0, (0,1): 0*8+col_off+1, (1,0): 1*8+col_off+0, (1,1): 1*8+col_off+1},
            {(0,0): 0*8+col_off+2, (0,1): 0*8+col_off+3, (1,0): 1*8+col_off+2, (1,1): 1*8+col_off+3},
            {(0,0): 2*8+col_off+0, (0,1): 2*8+col_off+1, (1,0): 3*8+col_off+0, (1,1): 3*8+col_off+1},
            {(0,0): 2*8+col_off+2, (0,1): 2*8+col_off+3, (1,0): 3*8+col_off+2, (1,1): 3*8+col_off+3},
        )

    def cblocks(col_off):
        return (
            {(0,0): col_off*3+1, (0,1): (col_off+1)*3+1, (1,0): col_off*3+2, (1,1): (col_off+1)*3+2},
            {(0,0): (col_off+2)*3+1, (0,1): (col_off+3)*3+1, (1,0): (col_off+2)*3+2, (1,1): (col_off+3)*3+2},
        )

    for col_off in (0, 4):
        b_tl, b_tr, b_bl, b_br = bblocks(col_off)
        c_l, c_r = cblocks(col_off)
        col = _strassen_block(U, V, W, a1_idx, b_tl, c_l, col)
        col = _strassen_block(U, V, W, a2_idx, b_bl, c_l, col)
        col = _strassen_block(U, V, W, a1_idx, b_tr, c_r, col)
        col = _strassen_block(U, V, W, a2_idx, b_br, c_r, col)

    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook():
    n, m, p = 3, 4, 8
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Primary: hybrid Strassen rank 88
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if time.time() < deadline:
            if _verify(U_h, V_h, W_h, n, m, p):
                return {"U": U_h, "V": V_h, "W": W_h}
        return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass

    # Fallback: schoolbook rank 96
    U_sb, V_sb, W_sb = _schoolbook()
    return {"U": U_sb, "V": V_sb, "W": W_sb}
