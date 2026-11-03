import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _build_strassen_2x2():
    """Build the 7 rank-1 terms for 2x2 Strassen."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (X00+X11)(Y00+Y11)
    U[0,0] = 1; U[1,1] = 1
    V[0,0] = 1; V[1,1] = 1
    W[0,0] = 1; W[1,1] = 1
    
    # M2 = (X10+X11)Y00
    U[1,2] = 1; U[1,3] = 1
    V[0,2] = 1
    W[1,0] = 1; W[1,1] = -1
    
    # M3 = X00(Y01-Y11)
    U[0,4] = 1
    V[0,4] = 1; V[1,4] = -1
    W[0,1] = 1; W[1,1] = 1
    
    # M4 = X11(Y10-Y00)
    U[1,5] = 1
    V[1,5] = 1; V[0,5] = -1
    W[0,0] = 1; W[1,0] = 1
    
    # M5 = (X00+X01)Y11
    U[0,6] = 1; U[0,7-1] = 1  # This is wrong, let me fix
    # Actually column 6 (0-indexed)
    U[0,6] = 1; U[0,6] += 0  # X00
    U[0,6] = 1; U[0,6] += 0  # This is getting confused
    
    # Let me restart with proper indexing
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (X00+X11)(Y00+Y11)
    U[0,0] = 1; U[1,1] = 1
    V[0,0] = 1; V[1,1] = 1
    W[0,0] = 1; W[1,1] = 1
    
    # M2 = (X10+X11)Y00
    U[1,2] = 1; U[1,3] = 1
    V[0,2] = 1
    W[1,0] = 1; W[1,1] = -1
    
    # M3 = X00(Y01-Y11)
    U[0,4] = 1
    V[0,4] = 1; V[1,4] = -1
    W[0,1] = 1; W[1,1] = 1
    
    # M4 = X11(Y10-Y00)
    U[1,5] = 1
    V[1,5] = 1; V[0,5] = -1
    W[0,0] = 1; W[1,0] = 1
    
    # M5 = (X00+X01)Y11
    U[0,6] = 1; U[0,6] = 1  # X00
    U[0,6] = 1; U[0,6] = 1  # X01... wait, X01 is position (0,1)
    # Let me be very careful. U has 4 rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # M5: (X00+X01)Y11
    U[0,6] = 1; U[1,6] = 1  # X00 and X01
    V[1,6] = 1  # Y11
    W[0,0] = -1; W[0,1] = 1
    
    # M6 = (X10-X00)(Y00+Y01)
    U[2,6] = 1; U[0,6] = -1  # X10 and X00
    V[0,6] = 1; V[1,6] = 1  # Y00 and Y01
    W[1,1] = 1
    
    # M7 = (X01-X11)(Y10+Y11)
    U[1,6] = 1; U[3,6] = -1  # X01 and X11
    V[1,6] = 1; V[2,6] = 1  # Y10 and Y11
    W[0,0] = 1
    
    # The above has errors in column 6. Let me redo properly.
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (X00+X11)(Y00+Y11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (X10+X11)Y00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,0] = 1; W[3,0] = -1
    
    # M3 = X00(Y01-Y11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[0,1] = 1; W[3,1] = 1
    
    # M4 = X11(Y10-Y00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,0] = 1; W[2,0] = 1
    
    # M5 = (X00+X01)Y11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,0] = -1; W[0,1] = 1
    
    # M6 = (X10-X00)(Y00+Y01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,1] = 1
    
    # M7 = (X01-X11)(Y10+Y11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,0] = 1
    
    return U, V, W

def _strassen_3x3_block():
    """
    Build the 7*7=49 rank-1 terms for multiplying two 3x3 block matrices
    where each block is a 2x2 scalar matrix, using Strassen at the block level.
    
    The 3x3 block Strassen decomposition uses 7 multiplications:
    Let A and B be 3x3 matrices of blocks (each 2x2).
    
    M1 = (A00 + A22)(B00 + B22)
    M2 = (A10 + A11 + A12)(B00 + B02)  -- wait, standard 3x3 Strassen needs care.
    
    Actually, let me use a known 3x3 Strassen-like decomposition.
    For 3x3, we can use:
    M1 = (A00 + A11 + A22)(B00 + B11 + B22)
    M2 = (A01 + A12 + A20)(B10 + B21 + B02)  -- cyclic
    ...
    
    This is getting complicated. Let me use a different approach.
    
    For 3x3 matrices, the standard approach is to use the fact that
    matrix multiplication is bilinear. I'll use a known rank-19 or similar
    decomposition for 3x3, or just use the schoolbook 27 and try to reduce.
    
    Actually, for the purpose of this task, let me just use the schoolbook
    27 multiplications for the 3x3 block level, giving total rank 27*4 = 108.
    But that's still better than 189.
    
    Wait, let me reconsider. The current program uses 2x2 blocks with Strassen
    (rank 7 per block pair), giving 3*3*3*7 = 189.
    
    If I use 3x3 blocks of size 1x1 (i.e., no sub-block structure), I get
    schoolbook 6*6*6 = 216, which is worse.
    
    If I use 2x2 blocks (3x3 of them) with Strassen, I get 189 as before.
    
    If I use 3x3 blocks (2x2 of them) with some decomposition...
    For 3x3 matrix multiplication, the best known rank is around 19-23.
    If I use rank 23 for 3x3, then 2*2*2*23 = 184, slightly better.
    If I use rank 19, then 2*2*2*19 = 152, much better!
    
    Let me find a rank-19 or similar decomposition for 3x3.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Use 2x2 blocks with Strassen (rank 7 per block triple)
    # Total rank: 3*3*3*7 = 189
    block_size = 2
    n_blocks = n // block_size  # 3
    
    rank = n_blocks ** 3 * 7  # 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)
    
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)
    
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)
    
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1 = (X00 + X11)(Y00 + Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M2 = (X10 + X11)Y00
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1
                
                # M3 = X00(Y01 - Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M4 = X11(Y10 - Y00)
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1
                
                # M5 = (X00 + X01)Y11
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1
                
                # M6 = (X10 - X00)(Y00 + Y01)
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M7 = (X01 - X11)(Y10 + Y11)
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # For now, return the 189-rank decomposition
    # Convert to integer if possible
    U_int = U.astype(int)
    V_int = V.astype(int)
    W_int = W.astype(int)
    
    # Verify integer representation
    if not np.allclose(U, U_int):
        U_int = U
    if not np.allclose(V, V_int):
        V_int = V
    if not np.allclose(W, W_int):
        W_int = W
    
    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
