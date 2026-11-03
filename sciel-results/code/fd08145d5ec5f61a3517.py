import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns list of (u, v, w) where each is a flat 4-element list.
    """
    terms = [
        # M1 = (A11+A22)(B11+B22)
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),
        # M2 = (A21+A22)B11
        ([0,0,1,1], [1,0,0,0], [0,0,0,1]),
        # M3 = A11(B21-B11)
        ([1,0,0,0], [0,1,0,0], [0,1,0,0]),
        # M4 = A22(B12-B22)
        ([0,0,0,1], [0,0,1,0], [0,0,1,0]),
        # M5 = (A11+A12)B22
        ([1,1,0,0], [0,0,0,1], [0,0,0,1]),
        # M6 = (A21-A11)(B11+B12)
        ([-1,0,1,0], [1,0,1,0], [0,1,0,0]),
        # M7 = (A12-A22)(B21+B22)
        ([0,1,0,-1], [0,1,1,1], [0,0,1,-1]),
    ]
    return terms

def _verify_decomp(U, V, W, n_tests=30, atol=1e-8):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    rng = np.random.default_rng(42)
    for _ in range(n_tests):
        A = rng.integers(-10, 10, (n, m))
        B = rng.integers(-10, 10, (m, p))
        C_expected = A @ B
        
        # Compute using tensor decomposition
        # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # Vectorized: for each r, outer product contribution
        C_computed = np.zeros((n, p), dtype=float)
        
        # Flatten A and B for dot products
        A_flat = A.flatten()  # (n*m,)
        B_flat = B.flatten()  # (m*p,)
        
        for r in range(rank):
            u_col = U_arr[:, r]  # (n*m,)
            v_col = V_arr[:, r]  # (m*p,)
            w_col = W_arr[:, r]  # (p*n,)
            
            scalar = np.dot(u_col, A_flat) * np.dot(v_col, B_flat)
            if abs(scalar) > 1e-15:
                C_computed += scalar * w_col.reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True

def _block_decomposition_356():
    """
    Block decomposition for 3x5x6 using split n=2+1, m=2+2+1, p=2+2+2.
    Uses Strassen rank-7 for 2x2x2 blocks where applicable.
    """
    n, m, p = DIMENSIONS
    strassen = _strassen_2x2x2()
    
    # Split indices
    # n: [0,1] and [2]
    # m: [0,1], [2,3], [4]
    # p: [0,1], [2,3], [4,5]
    
    blocks = [
        # (n_range, m_range, p_range)
        ((0,2), (0,2), (0,2)),  # 2x2x2 -> Strassen
        ((0,2), (0,2), (2,4)),  # 2x2x2 -> Strassen
        ((0,2), (0,2), (4,6)),  # 2x2x2 -> Strassen
        ((0,2), (2,4), (0,2)),  # 2x2x2 -> Strassen
        ((0,2), (2,4), (2,4)),  # 2x2x2 -> Strassen
        ((0,2), (2,4), (4,6)),  # 2x2x2 -> Strassen
        ((0,2), (4,5), (0,2)),  # 2x1x2 -> schoolbook rank 4
        ((0,2), (4,5), (2,4)),  # 2x1x2 -> schoolbook rank 4
        ((0,2), (4,5), (4,6)),  # 2x1x2 -> schoolbook rank 4
        ((2,3), (0,2), (0,2)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (0,2), (2,4)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (0,2), (4,6)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (2,4), (0,2)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (2,4), (2,4)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (2,4), (4,6)),  # 1x2x2 -> schoolbook rank 4
        ((2,3), (4,5), (0,2)),  # 1x1x2 -> schoolbook rank 2
        ((2,3), (4,5), (2,4)),  # 1x1x2 -> schoolbook rank 2
        ((2,3), (4,5), (4,6)),  # 1x1x2 -> schoolbook rank 2
    ]
    
    # Calculate total rank
    total_rank = 0
    for nr, mr, pr in blocks:
        ns = nr[1] - nr[0]
        ms = mr[1] - mr[0]
        ps = pr[1] - pr[0]
        if ns == 2 and ms == 2 and ps == 2:
            total_rank += 7
        else:
            total_rank += ns * ms * ps
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for nr, mr, pr in blocks:
        n0, n1 = nr
        m0, m1 = mr
        p0, p1 = pr
        ns = n1 - n0
        ms = m1 - m0
        ps = p1 - p0
        
        if ns == 2 and ms == 2 and ps == 2:
            # Use Strassen
            for u_flat, v_flat, w_flat in strassen:
                u_mat = np.array(u_flat, dtype=float).reshape(2, 2)
                v_mat = np.array(v_flat, dtype=float).reshape(2, 2)
                w_mat = np.array(w_flat, dtype=float).reshape(2, 2)
                
                for i in range(2):
                    for k in range(2):
                        U[(n0+i) * m + (m0+k)][col] += u_mat[i, k]
                for k in range(2):
                    for j in range(2):
                        V[(m0+k) * p + (p0+j)][col] += v_mat[k, j]
                for j in range(2):
                    for i in range(2):
                        W[(p0+j) * n + (n0+i)][col] += w_mat[i, j]
                col += 1
        else:
            # Schoolbook
            for i in range(ns):
                for k in range(ms):
                    for j in range(ps):
                        U[(n0+i) * m + (m0+k)][col] = 1
                        V[(m0+k) * p + (p0+j)][col] = 1
                        W[(p0+j) * n + (n0+i)][col] = 1
                        col += 1
    
    return U, V, W, total_rank

def _als_reduce(U, V, W, target_rank, deadline, max_iters=500):
    """
    Attempt to reduce rank using ALS (Alternating Least Squares).
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Target: reduce to target_rank columns.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    if target_rank >= rank:
        return U, V, W, rank
    
    # Initialize with first target_rank columns + random perturbation
    U_red = U_arr[:, :target_rank].copy()
    V_red = V_arr[:, :target_rank].copy()
    W_red = W_arr[:, :target_rank].copy()
    
    # Add small random perturbation to avoid degeneracy
    rng = np.random.default_rng(123)
    U_red += 0.01 * rng.standard_normal((n*m, target_rank))
    V_red += 0.01 * rng.standard_normal((m*p, target_rank))
    W_red += 0.01 * rng.standard_normal((p*n, target_rank))
    
    best_loss = float('inf')
    best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline - 30:
            break
        
        # Update U: for each row, solve least squares
        for i in range(n * m):
            # C[i, j] depends on U[i, :], V[:, :], W[:, :]
            # For fixed i, the contribution to each C entry is:
            # C[j*n + i] = sum_k sum_r U[i,r] * V[k*p+j, r] * W[j*n+i, r]
            
            # Build the linear system for U[i, :]
            # Let u = U[i, :] (target_rank,)
            # For each (k, j): C[j*n+i] = sum_r u[r] * (V[k*p+j, r] * W[j*n+i, r])
            # This is a linear system: A_u * u = b_u
            # where A_u[r1, r2] = sum_{k,j} V[k*p+j, r1] * W[j*n+i, r1] * V[k*p+j, r2] * W[j*n+i, r2]
            # b_u[r] = sum_{k,j} C_target[j*n+i] * V[k*p+j, r] * W[j*n+i, r]
            
            # But we don't have C_target directly. Instead, we use the current decomposition
            # to compute the "target" and then fit.
            
            # Actually, for ALS on tensor decomposition, we minimize:
            # sum over all (i,k,j) of (C_ijk - sum_r U[i,r]V[k,r]W[j,r])^2
            # where C_ijk is the true matrix multiplication tensor.
            
            # For fixed i, minimize over u = U[i,:]:
            # sum_{k,j} (C[i,k,j] - sum_r u[r] * V[k,r] * W[j,r])^2
            
            # Let's compute C[i,k,j] = sum_k' A[i,k'] * B[k',j] but we don't have A,B.
            # Instead, we use the fact that the true tensor is:
            # T[i,k,j] = 1 if i,k,j form a valid "diagonal" in some sense... 
            # No, the true tensor for matrix mult is: T[i,k,j] = delta_{i,j}... no.
            
            # The matrix multiplication tensor is:
            # T[i,k,j] such that C[i,j] = sum_k T[i,k,j] * A[i,k] * B[k,j]
            # So T[i,k,j] = 1 if the indices are "compatible" in the standard way.
            # Actually, T[i,k,j] = 1 for all valid (i,k,j) since C[i,j] = sum_k A[i,k]*B[k,j]
            
            # So the target tensor T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
            
            # For fixed i, we want to minimize:
            # sum_{k=0}^{m-1} sum_{j=0}^{p-1} (1 - sum_r u[r] * V[k*p+j, r] * W[j*n+i, r])^2
            
            # This is a least squares problem: minimize ||b - M*u||^2
            # where M is (m*p) x target_rank, b is (m*p,)
            # M[k*p+j, r] = V[k*p+j, r] * W[j*n+i, r]
            # b[k*p+j] = 1
            
            M = np.outer(V_red.flatten(), W_red[:, i].flatten())
            # Wait, that's not right. Let me be more careful.
            # M[row, r] where row = k*p+j
            # M[k*p+j, r] = V_red[k*p+j, r] * W_red[j*n+i, r]
            
            M = np.zeros((m * p, target_rank))
            for k in range(m):
                for j in range(p):
                    M[k*p+j, :] = V_red[k*p+j, :] * W_red[j*n+i, :]
            
            b = np.ones(m * p)
            
            # Solve M * u = b in least squares
            try:
                u_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            
            U_red[i, :] = u_sol
        
        # Similarly update V and W
        for k in range(m * p):
            M = np.zeros((n * p, target_rank))
            for i in range(n):
                for j in range(p):
                    M[i*p+j, :] = U_red[i*m+k, :] * W_red[j*n+i, :]
            b = np.ones(n * p)
            try:
                v_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            V_red[k, :] = v_sol
        
        for j in range(p * n):
            i = j % n  # Wait, W is indexed as W[j*n+i] where j is row of C, i is col of C
            # Actually W has shape (p*n, rank), indexed as W[j*n + i]
            # For fixed j (output row), we need to handle all i
            # Let me re-index: W_arr has rows indexed by (j, i) where j in [0,p), i in [0,n)
            # So row index = j*n + i
            # For fixed row r_w = j*n + i, we update W_red[r_w, :]
            # The constraint involves C[i, j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            # For fixed (j, i), minimize over w = W[j*n+i, :]:
            # sum_{k} (1 - sum_r U[i*m+k, r] * V[k*p+j, r] * w[r])^2
            # M[k, r] = U[i*m+k, r] * V[k*p+j, r]
            # b[k] = 1
            
            i = j % n
            j_actual = j // n
            
            M = np.zeros((m, target_rank))
            for k in range(m):
                M[k, :] = U_red[i*m+k, :] * V_red[k*p+j_actual, :]
            b = np.ones(m)
            try:
                w_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            W_red[j, :] = w_sol
        
        # Compute loss
        if iteration % 10 == 0:
            loss = 0.0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        pred = np.dot(U_red[i*m+k, :], V_red[k*p+j, :] * W_red[j*n+i, :])
                        loss += (1 - pred) ** 2
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
        
        if best_loss < 1e-6:
            break
    
    # Verify the best solution
    if best_loss < 1e-4:
        U_out = [[round(x, 12) for x in row] for row in best_U]
        V_out = [[round(x, 12) for x in row] for row in best_V]
        W_out = [[round(x, 12) for x in row] for row in best_W]
        
        if _verify_decomp(U_out, V_out, W_out, n_tests=10, atol=1e-6):
            return U_out, V_out, W_out, target_rank
    
    return U, V, W, rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Get block decomposition
    U, V, W, rank = _block_decomposition_356()
    
    # Verify the initial decomposition
    if not _verify_decomp(U, V, W, n_tests=10, atol=1e-8):
        # Fallback to full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    for target_rank in range(rank - 1, max(rank - 20, 0), -1):
        if time.time() > deadline - 40:
            break
        
        U_try, V_try, W_try, achieved_rank = _als_reduce(
            best_U, best_V, best_W, target_rank, deadline, max_iters=200
        )
        
        if achieved_rank < best_rank:
            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, achieved_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
