import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s limit
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T[u, v, w] = 1 if (u, v, w) is a valid product index
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k]
    # Here u = i (0..n-1), v = j (0..m-1), w = k (0..p-1)
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()  # size n*m*p = 36
    
    def verify(U, V, W):
        """Check if U, V, W form an exact decomposition with half-integer entries."""
        rank = U.shape[1]
        # Compute the tensor from U, V, W
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            # U[:, r] is size n*m, V[:, r] is size m*p, W[:, r] is size p*n
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            # The contribution to T[i, j, k] is sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Let's compute this properly
            for i in range(n):
                for j in range(m):
                    u_val = U[i * m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j * p + k, r]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i, r]
                        tensor[i, j, k] += u_val * v_val * w_val
        # Check if it matches T exactly (with half-integer tolerance)
        if not np.allclose(tensor, T, atol=1e-6):
            return False
        # Check half-integer constraint: all entries must be multiples of 0.5
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-6):
                return False
        return True
    
    def snap_to_halfint(M):
        """Snap matrix entries to nearest half-integer."""
        return np.round(M * 2) / 2.0
    
    def als_decompose(rank, num_iters=30, seed=0):
        """ALS decomposition for the matrix multiplication tensor."""
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W with random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        for it in range(num_iters):
            # Update U
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed (i,j), we want to find U[i*m+j, :] such that
            # T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
            # Let P_r = V[:, r] * W[:, r] element-wise, but indexed differently
            # Actually, for fixed (i,j), the coefficient of U[i*m+j, r] is:
            # sum_k ... no wait.
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So for fixed (i,j), we have a vector t_{ij} of length p (over k)
            # and for each r, the coefficient vector c_{r, ij} has component k: V[j*p+k, r] * W[k*n+i, r]
            # We want sum_r U[i*m+j, r] * c_{r, ij} = t_{ij}
            # This is a least squares problem: for each (i,j), solve A_{ij} U[i*m+j, :] = t_{ij}
            # where A_{ij} is p x rank, with A_{ij}[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build A: p x rank
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Target: t of length p
                    t_vec = np.array([T[i, j, k] for k in range(p)])
                    # Solve least squares: A_mat U_col = t_vec
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            # For fixed (j,k), coefficient of V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
            # T[i,j,k] = sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r])
            # For fixed (j,k), we have a vector t_{jk} of length n (over i)
            # A_{jk}[i, r] = U[i*m+j, r] * W[k*n+i, r]
            
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    t_vec = np.array([T[i, j, k] for i in range(n)])
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            # For fixed (k,i), coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
            # T[i,j,k] = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
            # For fixed (k,i), we have a vector t_{ki} of length m (over j)
            # A_{ki}[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    t_vec = np.array([T[i, j, k] for j in range(m)])
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
        
        # Snap to half-integers
        U_snapped = snap_to_halfint(U)
        V_snapped = snap_to_halfint(V)
        W_snapped = snap_to_halfint(W)
        
        # Verify
        if verify(U_snapped, V_snapped, W_snapped):
            return U_snapped, V_snapped, W_snapped, True
        
        # Try without snapping first (maybe the solution is exact in floating point)
        if verify(U, V, W):
            return U, V, W, True
        
        return U_snapped, V_snapped, W_snapped, False
    
    # Try increasing ranks from the theoretical minimum
    # Theoretical lower bound for matmul(n, m, p) is at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But for exact decompositions with integer/half-integer entries, we need to search.
    # Known: rank of matmul(3,3,3) is at most 23, and for (3,3,4) likely around 20-25.
    # Let's try ranks from 12 up to 36 (schoolbook)
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Schoolbook solution
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 12 to best_rank-1
    for rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        # Multiple restarts for this rank
        found = False
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            U, V, W, success = als_decompose(rank, num_iters=20, seed=rank * 100 + restart)
            if success:
                found = True
                best_result = (U, V, W, rank)
                break
        
        if found:
            break
        # If we've tried several ranks without success, we might want to increase restarts
        # but we're constrained by time
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to lists for the return format
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
