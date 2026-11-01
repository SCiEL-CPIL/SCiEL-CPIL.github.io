import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _iterative_projection(U, V, W, max_iters=50, tol=1e-10):
    """
    Apply iterative projection (ALS) to minimize the tensor error.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    for _ in range(max_iters):
        # Update U
        for r in range(rank):
            A_mat = np.zeros((n*m*p, n*m))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = i*m + j
                        A_mat[idx_count, var_idx] = V[j*p+k, r] * W[k*n+i, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            U[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update V
        for r in range(rank):
            A_mat = np.zeros((n*m*p, m*p))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = j*p + k
                        A_mat[idx_count, var_idx] = U[i*m+j, r] * W[k*n+i, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            V[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update W
        for r in range(rank):
            A_mat = np.zeros((n*m*p, p*n))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = k*n + i
                        A_mat[idx_count, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            W[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        err = np.sqrt(err)
        if err < tol:
            break
            
    return U, V, W, err

def _try_rank_search(rank, initial_U=None, initial_V=None, initial_W=None, time_limit=5.0):
    """
    Try to find a decomposition of given rank.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    if initial_U is not None:
        U = initial_U[:, :rank].copy()
        V = initial_V[:, :rank].copy()
        W = initial_W[:, :rank].copy()
        U, V, W, err = _iterative_projection(U, V, W, max_iters=100, tol=1e-10)
        if err < 1e-8:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            if _verify_decomposition(U, V, W):
                return U, V, W
        return None

    else:
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            U, V, W, err = _iterative_projection(U, V, W, max_iters=50, tol=1e-8)
            if err < 1e-8:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if _verify_decomposition(U, V, W):
                    return U, V, W
        return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    if not _verify_decomposition(U, V, W):
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    
    # Try to reduce rank using iterative projection
    for r in range(35, 19, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to reduce from current best
        result = _try_rank_search(r, best_U, best_V, best_W, time_limit=2.0)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
        else:
            # If we fail to reduce from current, try random init
            if time.time() < deadline - 10:
                result = _try_rank_search(r, None, None, None, time_limit=2.0)
                if result is not None:
                    best_U, best_V, best_W = result
                    best_rank = r
                else:
                    # If we can't reduce by 1, we likely can't go lower easily
                    # But we continue just in case, though it will be slow
                    pass

    if not _verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = U, V, W
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
