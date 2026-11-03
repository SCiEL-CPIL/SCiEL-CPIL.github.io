import time
import numpy as np
from itertools import product

DEADLINE_S = 140.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U,V,W) for rank-49 Strassen decomposition of 4x4x4 matrix multiplication."""
    s2U, s2V, s2W = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    strassen_ops = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1, 'C01': -1, 'C10': -1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B10': 1, 'B11': -1}, {'C01': 1, 'C11': -1}),
        ({'A11': 1}, {'B01': 1, 'B00': -1}, {'C00': -1, 'C01': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': 1, 'C11': -1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B01': 1, 'B11': 1}, {'C00': -1, 'C10': 1}),
    ]
    
    block_U = {
        'A00': (0, 4),
        'A01': (2, 6),
        'A10': (4, 8),
        'A11': (6, 10),
    }
    block_V = {
        'B00': (0, 4),
        'B01': (2, 6),
        'B10': (4, 8),
        'B11': (6, 10),
    }
    block_W = {
        'C00': (0, 4),
        'C01': (2, 6),
        'C10': (4, 8),
        'C11': (6, 10),
    }
    
    for op_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_ops):
        base_r = op_idx * 7
        for r2 in range(7):
            r = base_r + r2
            for blk, c in u_blocks.items():
                s, e = block_U[blk]
                for i2 in range(2):
                    for j2 in range(2):
                        U[s + i2*2 + j2, r] += c * s2U[i2*2 + j2, r2]
            
            for blk, c in v_blocks.items():
                s, e = block_V[blk]
                for j2 in range(2):
                    for k2 in range(2):
                        V[s + j2*2 + k2, r] += c * s2V[j2*2 + k2, r2]
            
            for blk, c in w_blocks.items():
                s, e = block_W[blk]
                for k2 in range(2):
                    for i2 in range(2):
                        W[s + k2*2 + i2, r] += c * s2W[k2*2 + i2, r2]
    
    return U, V, W


def _make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off, n, m, p):
    """Create schoolbook for a sub-block embedded in the full space."""
    rank = n_b * m_b * p_b
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[(i_off+i)*m + (j_off+j), c] = 1.0
                V[(j_off+j)*p + (k_off+k), c] = 1.0
                W[(k_off+k)*n + (i_off+i), c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps, n, m, p):
    """Combine a list of (U,V,W) decompositions into one."""
    if not decomps:
        return None
    
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def _try_remove_column(U, V, W, r_idx, n, m, p, tol=1e-6):
    """Check if removing column r_idx still gives a valid decomposition."""
    R = U.shape[1]
    mask = [i for i in range(R) if i != r_idx]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    return _validate(U_new, V_new, W_new, tol=tol), (U_new, V_new, W_new)


def _peel_aggressive(U, V, W, n, m, p, time_limit=30.0, max_rounds=200):
    """Iteratively try to remove columns, using a smarter ordering."""
    t0 = time.time()
    U_cur, V_cur, W_cur = U, V, W
    rank_cur = U_cur.shape[1]
    
    # Try columns in order of "importance" - columns with smaller norms first
    # as they're more likely to be removable
    def col_norms():
        U_norms = np.sum(U_cur**2, axis=0)
        V_norms = np.sum(V_cur**2, axis=0)
        W_norms = np.sum(W_cur**2, axis=0)
        return (U_norms + V_norms + W_norms) / 3.0
    
    for round_num in range(max_rounds):
        if time.time() - t0 > time_limit:
            break
        if rank_cur < 5:
            break
        
        norms = col_norms()
        # Sort columns by norm, try smallest first
        order = np.argsort(norms)
        
        removed = False
        for r_idx in order:
            if time.time() - t0 > time_limit:
                break
            
            valid, (U_new, V_new, W_new) = _try_remove_column(U_cur, V_cur, W_cur, r_idx, n, m, p, tol=1e-5)
            if valid:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                rank_cur -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U_cur, V_cur, W_cur


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Split the inner dimension (5) into 1+4.
    # A is 4x5, B is 5x5, C is 4x5.
    # A = [A1 (4x1) | A2 (4x4)], B = [[B11 (1x1), B12 (1x4)], [B21 (4x1), B22 (4x4)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # A1*B11: 4x1 * 1x1, schoolbook rank 4
    # A1*B12: 4x1 * 1x4, schoolbook rank 16
    # A2*B21: 4x4 * 4x1, schoolbook rank 16
    # A2*B22: 4x4 * 4x4, use Strassen rank 49
    #
    # Total: 4 + 16 + 16 + 49 = 85
    
    decomps = []
    
    # A1*B11: A[0:4,0:1] * B[0:1,0:1] -> C[0:4,0:1]
    decomps.append(_make_schoolbook(4, 1, 1, 0, 0, 0, n, m, p))
    
    # A1*B12: A[0:4,0:1] * B[0:1,1:5] -> C[0:4,1:5]
    decomps.append(_make_schoolbook(4, 1, 4, 0, 0, 1, n, m, p))
    
    # A2*B21: A[0:4,1:5] * B[1:5,0:1] -> C[0:4,0:1]
    decomps.append(_make_schoolbook(4, 4, 1, 0, 1, 0, n, m, p))
    
    # A2*B22: A[0:4,1:5] * B[1:5,1:5] -> C[0:4,1:5]
    # Use Strassen rank 49 for 4x4x4
    s4U, s4V, s4W = _strassen_4x4x4()
    
    U_embed = np.zeros((n*m, 49))
    V_embed = np.zeros((m*p, 49))
    W_embed = np.zeros((p*n, 49))
    
    for i in range(4):
        for j in range(4):
            U_embed[i*5 + (1+j), :] = s4U[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V_embed[(1+j)*5 + (1+k), :] = s4V[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_embed[(1+k)*4 + i, :] = s4W[k*4+i, :]
    
    decomps.append((U_embed, V_embed, W_embed))
    
    # Combine all decompositions
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via aggressive peeling
    remaining_time = DEADLINE_S - 15 - (time.time() - t0)
    if remaining_time > 5:
        U_red, V_red, W_red = _peel_aggressive(U, V, W, n, m, p, time_limit=remaining_time, max_rounds=200)
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
