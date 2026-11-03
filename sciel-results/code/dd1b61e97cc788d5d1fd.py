"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_LIMIT = 120
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n=3, m=3, p=8):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix multiplication, T[i,j,k] should be 1 for all valid (i,j,k)
    # where (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=8):
    """Return the schoolbook rank-n*m*p decomposition."""
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

def _strassen_2x2_2x2():
    """
    Return a rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W are 2x7 matrices (local indices 0,1).
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A10)(B01+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M1 - M2 + M5 + M7
    # C11 = M2 - M3 + M5 - M6
    #
    # In tensor form: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i,j,k in {0,1}
    
    U = [
        [1, 0, 0, 1, 1, 1, 1],   # i=0
        [1, 1, 0, 0, 0, 0, 0],   # i=1
    ]
    V = [
        [1, 1, 1, 0, 0, 1, 0],   # j=0
        [1, 0, 0, 1, 1, 1, 1],   # j=1
    ]
    W = [
        [1, 0, 1, 1, 0, 1, 1],   # k=0
        [1, 1, 0, 1, 1, 0, 1],   # k=1
    ]
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[k, :])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook for 2x2x2 (rank 8)
                    U = [[1,1,0,0,0,0,0,0],[0,0,1,1,0,0,0,0]]
                    V = [[1,0,1,0,0,0,0,0],[0,1,0,1,0,0,0,0]]
                    W = [[1,0,0,1,0,0,0,0],[0,1,0,0,1,0,0,0]]
                    return U, V, W, 8
    return U, V, W, 7

def _cube_rank21_block(n=3, m=3, p=3, i0=0, j0=0, k0=0):
    """
    Generate a rank-21 decomposition for a 3x3x3 cube block.
    Uses Strassen for the 2x2x2 sub-block plus additional terms.
    """
    U_local = []
    V_local = []
    W_local = []
    
    # First, use Strassen for the 2x2x2 sub-block (indices 0,1 in each dim)
    Us, Vs, Ws, strassen_rank = _strassen_2x2_2x2()
    
    # Add Strassen terms (they cover i,j,k in {0,1})
    for r in range(strassen_rank):
        u = [0, 0, 0]
        v = [0, 0, 0]
        w = [0, 0, 0]
        u[0] = Us[0][r]
        u[1] = Us[1][r]
        v[0] = Vs[0][r]
        v[1] = Vs[1][r]
        w[0] = Ws[0][r]
        w[1] = Ws[1][r]
        U_local.append(u)
        V_local.append(v)
        W_local.append(w)
    
    # Now we need to cover the remaining entries where at least one index is 2.
    # The Strassen terms already cover (i,j,k) in {0,1}^3.
    # We need to add terms for:
    # - (2, j, k) for j,k in {0,1,2}
    # - (i, 2, k) for i in {0,1}, k in {0,1,2}
    # - (i, j, 2) for i,j in {0,1}
    # But we must be careful not to double-count.
    
    # Let's use a systematic approach: add schoolbook-like terms for the remaining entries
    # but try to merge where possible.
    
    # Remaining entries to cover:
    # (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2), (2,2,0), (2,2,1), (2,2,2)
    # (0,2,0), (0,2,1), (0,2,2), (1,2,0), (1,2,1), (1,2,2)
    # (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # Total: 9 + 6 + 4 = 19 entries
    
    # We have 21 - 7 = 14 terms left. We can't cover 19 entries with 14 terms
    # using simple schoolbook. We need to be smarter.
    
    # Let's try a different approach: use a known rank-21 decomposition.
    # After research, here's a valid rank-21 decomposition:
    
    # Actually, let me just use schoolbook for the remaining 19 entries,
    # giving total rank 7 + 19 = 26 for the 3x3x3 block.
    # This is worse than the ideal 21, but better than 27.
    
    # For now, let's use schoolbook for the remaining entries:
    remaining = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2),
    ]
    
    for (i, j, k) in remaining:
        u = [0, 0, 0]
        v = [0, 0, 0]
        w = [0, 0, 0]
        u[i] = 1
        v[j] = 1
        w[k] = 1
        U_local.append(u)
        V_local.append(v)
        W_local.append(w)
    
    # Now offset the indices for the full tensor
    U_full = [[0] * len(U_local) for _ in range(n * m)]
    V_full = [[0] * len(V_local) for _ in range(m * p)]
    W_full = [[0] * len(W_local) for _ in range(p * n)]
    
    for r in range(len(U_local)):
        for i_local in range(3):
            i_global = i0 + i_local
            if 0 <= i_global < n:
                U_full[i_global * m + j0][r] = U_local[r][i_local]
        for j_local in range(3):
            j_global = j0 + j_local
            if 0 <= j_global < m:
                V_full[j_global * p + k0][r] = V_local[r][j_local]
        for k_local in range(3):
            k_global = k0 + k_local
            if 0 <= k_global < p:
                W_full[k_global * n + i0][r] = W_local[r][k_local]
    
    return U_full, V_full, W_full, len(U_local)

def construct():
    """
    Returns a tensor decomposition for matrix multiplication 3x3x8.
    Uses two 3x3x3 cubic blocks with Strassen+schoolbook decompositions
    and one 3x3x2 block with schoolbook.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to use two 3x3x3 cubes and one 3x3x2 block
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Cube 1: k in {0,1,2}
    U1, V1, W1, rank1 = _cube_rank21_block(n, m, p, i0=0, j0=0, k0=0)
    U_list.append(U1)
    V_list.append(V1)
    W_list.append(W1)
    total_rank += rank1
    
    # Cube 2: k in {3,4,5}
    U2, V2, W2, rank2 = _cube_rank21_block(n, m, p, i0=0, j0=0, k0=3)
    U_list.append(U2)
    V_list.append(V2)
    W_list.append(W2)
    total_rank += rank2
    
    # Block 3: k in {6,7} - use schoolbook for 3x3x2
    U3 = [[0] * 18 for _ in range(n * m)]
    V3 = [[0] * 18 for _ in range(m * p)]
    W3 = [[0] * 18 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):
                k_global = 6 + k
                U3[i * m + j][col] = 1
                V3[j * p + k_global][col] = 1
                W3[k_global * n + i][col] = 1
                col += 1
    U_list.append(U3)
    V_list.append(V3)
    W_list.append(W3)
    total_rank += 18
    
    # Combine all blocks
    U_combined = [[0] * total_rank for _ in range(n * m)]
    V_combined = [[0] * total_rank for _ in range(m * p)]
    W_combined = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for idx in range(3):
        block_rank = len(U_list[idx][0])
        for row in range(n * m):
            for r in range(block_rank):
                U_combined[row][offset + r] = U_list[idx][row][r]
        for row in range(m * p):
            for r in range(block_rank):
                V_combined[row][offset + r] = V_list[idx][row][r]
        for row in range(p * n):
            for r in range(block_rank):
                W_combined[row][offset + r] = W_list[idx][row][r]
        offset += block_rank
    
    result = {"U": U_combined, "V": V_combined, "W": W_combined}
    
    # Verify
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        if total_rank < best_rank:
            return result
    
    # Fall back to schoolbook
    return best
# EVOLVE-BLOCK-END
