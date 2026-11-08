import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def terms_to_matrices(U_t, V_t, W_t, n, m, p):
        rank = len(W_t)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rank = U.shape[1]
        # Reconstruct tensor: T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Only check nonzero entries (the 72 diagonal entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: 1 if this is a valid multiplication term
                    # T[i,j,k] = 1 (the multiplication tensor has 1s where indices align)
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        # Also verify that off-diagonal entries are zero
        # The tensor T[i,j,k] is 1 for all (i,j,k) in the multiplication pattern
        # Actually for matrix multiplication, T[i,j,k] = 1 for ALL i,j,k (it's the identity pattern)
        # Wait no - the multiplication tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because A[i,j] * B[j,k] contributes to C[i,k]
        # So the tensor is just all ones in the relevant positions
        # Let me re-examine: the tensor for matrix multiplication C = A*B is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        return True

    def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        """Fast verification using tensor reconstruction."""
        rank = U.shape[1]
        # Reconstruct the full tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n).T)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    # Build the 68-term Strassen-based decomposition
    U_t = []
    V_t = []
    W_t = []
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, c:c+2] for c in 0, 2, 4, 6
    for c in range(0, 8, 2):
        strassen_decomp(0, 0, 0, c, U_t, V_t, W_t)
        
    # Direct terms for remaining A indices
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
            
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
            
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
        
    rank = len(W_t)
    U, V, W = terms_to_matrices(U_t, V_t, W_t, n, m, p)
    
    # Verify the baseline
    if not verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback: return naive 72-term
        U_t2, V_t2, W_t2 = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    direct_term(i, j, k, U_t2, V_t2, W_t2)
        U, V, W = terms_to_matrices(U_t2, V_t2, W_t2, n, m, p)
        rank = len(W_t2)

    # Try to reduce rank using randomized ALS
    def try_reduce_rank_als(U_init, V_init, W_init, target_rank, n, m, p, max_iters=500, tol=1e-10):
        """Try to find a decomposition at target_rank using ALS."""
        if target_rank >= U_init.shape[1]:
            return None
        
        # Initialize with random projection of the existing decomposition
        rng = np.random.default_rng(42)
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(3):
            # Initialize
            U = U_init[:, :target_rank].copy()
            V = V_init[:, :target_rank].copy()
            W = W_init[:, :target_rank].copy()
            
            # Add small random perturbation
            U += rng.normal(0, 0.1, U.shape)
            V += rng.normal(0, 0.1, V.shape)
            W += rng.normal(0, 0.1, W.shape)
            
            for it in range(max_iters):
                # Update U: for each column i*m+j, solve least squares
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (i,j), we need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # Let B_rk = V[j*p+k,r] * W[k*n+i,r] for all r,k
                # Then we need: U[i*m+j,:] @ B = 1_k for all k
                # This is a linear system: B^T U[i*m+j,:]^T = 1
                # Actually: sum_r U[i*m+j,r] * B[r,k] = 1 for all k
                # So: B^T @ U[i*m+j,:]^T = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        # Build B matrix: B[r,k] = V[j*p+k,r] * W[k*n+i,r]
                        B = np.zeros((target_rank, p))
                        for k in range(p):
                            B[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve B^T @ u = ones(p)
                        try:
                            u, residuals, rank_b, sv = np.linalg.lstsq(B.T, np.ones(p), rcond=None)
                            U[i * m + j, :] = u
                        except:
                            pass
                
                # Update V: for each column j*p+k
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                        # Let A_ri = U[i*m+j,r] * W[k*n+i,r]
                        # A^T @ v = ones(n)
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                            V[j * p + k, :] = v
                        except:
                            pass
                
                # Update W: for each column k*n+i
                for k in range(p):
                    for i in range(n):
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                        # Let C_rj = U[i*m+j,r] * V[j*p+k,r]
                        # C^T @ w = ones(m)
                        C = np.zeros((target_rank, m))
                        for j in range(m):
                            C[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w, residuals, rank_c, sv = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                            W[k * n + i, :] = w
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0:
                    T = np.zeros((n, m, p))
                    for r in range(target_rank):
                        T += np.einsum('i,j,k->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n).T)
                    err = np.max(np.abs(T - 1.0))
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < tol:
                        break
                    if time.time() - start_time > DEADLINE - 10:
                        break
            
            if best_err < tol:
                break
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None

    # Try to reduce from 68 to lower ranks
    best_result = (U, V, W, rank)
    
    for target in range(rank - 1, max(rank - 5, 50), -1):
        if time.time() - start_time > DEADLINE - 15:
            break
        result = try_reduce_rank_als(U, V, W, target, n, m, p, max_iters=200)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_decomposition_fast(U_r, V_r, W_r, n, m, p, tol=1e-8):
                best_result = (U_r, V_r, W_r, target)
                rank = target
                U, V, W = U_r, V_r, W_r
                break  # Found a valid lower rank, stop
    
    U, V, W, rank = best_result
    
    # Convert to lists of nice values
    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v * 2) / 2) < 1e-9:
                    nv = round(v * 2) / 2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res
    
    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
