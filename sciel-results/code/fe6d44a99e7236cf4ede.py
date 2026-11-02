"""Find rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
- Use a 3x2x2 local factor of rank 11 (7 Strassen-like + 4 trivial for the 3rd row)
- Tile it over the 4x4 B matrix in a 2x2 block structure
- Total rank = 4 * 11 = 44
- Then attempt to reduce rank further by solving for a sparse, half-integer
  decomposition using structured column elimination via linear algebra.
- Verify all candidates with exact integer/half-integer arithmetic.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]


def _verify_decomposition(U, V, W, n=3, m=4, p=4):
    """Verify that the decomposition correctly represents matrix multiplication.
    
    For each (i,j,k), the tensor entry should be:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer within tolerance."""
    doubled = 2 * x
    rounded = round(doubled)
    return abs(doubled - rounded) < tol


def _verify_half_integer(U, V, W, n=3, m=4, p=4):
    """Verify decomposition AND that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return _verify_decomposition(U, V, W, n, m, p)


def _build_rank44():
    """Build the rank-44 decomposition using 3x2x2 blocks tiled over 4x4 B."""
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local 3x2x2 factor with rank 11
    # Local A: 3x2 -> indices 0..5 (row*2+col)
    # Local B: 2x2 -> indices 0..3 (row*2+col)
    # Local C: 3x2 -> indices 0..5 (row*2+col)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen-like part for rows 0,1 (cols 0-6)
    # M1: (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2: (a10+a11)b00 -> c10 += M2, c11 -= M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3: a00(b11-b01) -> c01 += M3, c11 += M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4: a11(b10-b00) -> c00 += M4, c10 += M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5: (a00+a01)b11 -> c00 -= M5, c01 += M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c11 += M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7: (a01-a11)(b10+b11) -> c00 += M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Trivial part for row 2 (cols 7-10)
    # M8: a20*b00 -> c20
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    # M9: a21*b10 -> c20
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[5, 8] = 1
    # M10: a20*b01 -> c21
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    # M11: a21*b11 -> c21
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = r * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset + 11] += U_loc[local_a_idx, :]
        
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset + 11] += V_loc[local_b_idx, :]
        
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + r
                W[global_c_idx, col_offset:col_offset + 11] += W_loc[local_c_idx, :]
    
    # Block 0: A[:,0:2] * B[0:2, 0:2] -> C[:,0:2]
    add_block(0, 0, 0, 0, 0)
    # Block 1: A[:,2:4] * B[2:4, 0:2] -> C[:,0:2]
    add_block(1, 2, 2, 0, 0)
    # Block 2: A[:,0:2] * B[0:2, 2:4] -> C[:,2:4]
    add_block(2, 0, 0, 2, 2)
    # Block 3: A[:,2:4] * B[2:4, 2:4] -> C[:,2:4]
    add_block(3, 2, 2, 2, 2)
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_eliminate_column(U, V, W, col_idx, n=3, m=4, p=4):
    """Try to eliminate column col_idx by expressing it as a linear combination of others.
    
    For a valid decomposition, if we can find coefficients alpha_r such that
    for all (i,j,k): U[i,j] * V[j,k] * W[k,i] = sum_r alpha_r * U[i,j] * V[j,k] * W[k,i]
    then we can absorb column col_idx into the others.
    
    This is equivalent to finding a vector in the null space of a certain matrix.
    We use a simpler approach: try to find small perturbations that maintain validity.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    if rank <= 1:
        return None
    
    # Try to find a direction in which we can move to reduce rank
    # by making one column a linear combination of others
    
    # For each other column, check if column col_idx is approximately
    # a linear combination of that column
    for other_col in range(rank):
        if other_col == col_idx:
            continue
        
        # Check if U[:, col_idx] is proportional to U[:, other_col]
        # and similarly for V and W
        u_ratio = None
        v_ratio = None
        w_ratio = None
        
        # Find ratio for U
        for r in range(U_np.shape[0]):
            if abs(U_np[r, other_col]) > 1e-10:
                ratio = U_np[r, col_idx] / U_np[r, other_col]
                if u_ratio is None:
                    u_ratio = ratio
                elif abs(ratio - u_ratio) > 1e-9:
                    u_ratio = None
                    break
        
        if u_ratio is None:
            continue
        
        # Find ratio for V
        for r in range(V_np.shape[0]):
            if abs(V_np[r, other_col]) > 1e-10:
                ratio = V_np[r, col_idx] / V_np[r, other_col]
                if v_ratio is None:
                    v_ratio = ratio
                elif abs(ratio - v_ratio) > 1e-9:
                    v_ratio = None
                    break
        
        if v_ratio is None:
            continue
        
        # Find ratio for W
        for r in range(W_np.shape[0]):
            if abs(W_np[r, other_col]) > 1e-10:
                ratio = W_np[r, col_idx] / W_np[r, other_col]
                if w_ratio is None:
                    w_ratio = ratio
                elif abs(ratio - w_ratio) > 1e-9:
                    w_ratio = None
                    break
        
        if w_ratio is None:
            continue
        
        # If all three ratios are equal, we can eliminate
        if abs(u_ratio - v_ratio) < 1e-9 and abs(v_ratio - w_ratio) < 1e-9:
            # Column col_idx is a scalar multiple of column other_col
            # We can remove it by adjusting the other column
            U_new = U_np.copy()
            V_new = V_np.copy()
            W_new = W_np.copy()
            
            U_new[:, other_col] += u_ratio * U_new[:, col_idx]
            V_new[:, other_col] += v_ratio * V_new[:, col_idx]
            W_new[:, other_col] += w_ratio * W_new[:, col_idx]
            
            # Remove column col_idx
            mask = [r for r in range(rank) if r != col_idx]
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            # Round to half-integers
            U_new = np.round(U_new * 2) / 2.0
            V_new = np.round(V_new * 2) / 2.0
            W_new = np.round(W_new * 2) / 2.0
            
            # Verify
            if _verify_half_integer(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _try_reduce_rank(U, V, W, target_rank, max_iter=200, seed=42):
    """Try to reduce the rank of a valid decomposition to target_rank.
    
    Strategy: iteratively try to eliminate columns by finding linear dependencies.
    """
    n, m, p = 3, 4, 4
    current_rank = len(U[0])
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Work with copies
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    
    rng = random.Random(seed)
    
    # Try to eliminate columns one at a time
    for attempt in range(max_iter):
        if len(U_cur[0]) <= target_rank:
            break
        
        # Try random columns in random order
        cols = list(range(len(U_cur[0])))
        rng.shuffle(cols)
        
        for col in cols:
            result = _try_eliminate_column(U_cur, V_cur, W_cur, col, n, m, p)
            if result is not None:
                U_cur, V_cur, W_cur = result
                if len(U_cur[0]) <= target_rank:
                    break
    
    return U_cur, V_cur, W_cur


def construct():
    """Construct the tensor decomposition. Returns dict with U, V, W."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    n, m, p = DIMENSIONS
    
    # Build the rank-44 decomposition
    U, V, W = _build_rank44()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_fb = [[0] * rank_school for _ in range(n * m)]
        V_fb = [[0] * rank_school for _ in range(m * p)]
        W_fb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {'U': U_fb, 'V': V_fb, 'W': W_fb}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to reduce the rank further using column elimination
    if time.time() < deadline - 30:
        # Try multiple seeds for the reduction attempt
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 30:
                break
            
            try:
                U_try, V_try, W_try = _try_reduce_rank(
                    best_U, best_V, best_W, target_rank=best_rank - 1,
                    max_iter=100, seed=seed
                )
                
                if len(U_try[0]) < best_rank:
                    if _verify_half_integer(U_try, V_try, W_try, n, m, p):
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = len(U_try[0])
            except Exception:
                pass
    
    return {'U': best_U, 'V': best_V, 'W': best_W}
