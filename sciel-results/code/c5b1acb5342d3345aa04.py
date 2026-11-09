"""Matrix multiplication 3x4x5 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 5

    def verify_solution(U, V, W, r, tol=1e-10):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    err = abs(val - 1.0)
                    if err > tol:
                        return False
        return True

    def verify_fast(U, V, W, r, tol=1e-10):
        """Vectorized verification using matrix operations."""
        # For each (i,j), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This can be done by checking: for each r, U[:,r] @ (V * W_outer) should give ones
        # Actually, let's just do it efficiently with broadcasting
        for r in range(r):
            # For each r, we need: sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed r: U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            u_col = U[:, r]  # shape (n*m,)
            v_col = V[:, r]  # shape (m*p,)
            w_col = W[:, r]  # shape (p*n,)
            # Reshape to (n, m), (m, p), (p, n)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            # The contribution to the tensor from rank-r term is:
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # We need sum_r of this to be 1 for all (i,j,k)
            # This is equivalent to: for each r, the "outer product" u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
            # summed over r equals 1
        # Just do the direct check but vectorized over (i,j,k)
        # Build the full tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a contraction over r
        # We can compute this as: for each (i,j,k), sum_r ...
        # Efficient way: einsum
        T = np.einsum('ijr,jkr,ki->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def try_rank_als(r, seed=42, max_iters=20000):
        """Try to find a rank-r decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        prev_residual = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # This is: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then: M @ U[i*m+j, :] = ones(p)
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(r):
                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        residual += (val - 1.0) ** 2
            
            if residual < 1e-12:
                break
            
            if residual < prev_residual * (1 - 1e-6):
                prev_residual = residual
            else:
                # Stagnation, try a small perturbation
                if it > 10:
                    U += rng.randn(n * m, r) * 0.01
                    V += rng.randn(m * p, r) * 0.01
                    W += rng.randn(p * n, r) * 0.01
        
        # Final check
        if verify_solution(U, V, W, r):
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_solution(U_rounded, V_rounded, W_rounded, r):
                return U_rounded, V_rounded, W_rounded, True
            else:
                return U, V, W, True
        
        return None, None, None, False

    # The known optimal rank for 3x4x5 matrix multiplication is 16
    # Try multiple seeds for rank 16
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        U, V, W, success = try_rank_als(16, seed=seed, max_iters=5000)
        if success:
            U_list = [[float(U[i][j]) for j in range(16)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(16)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(16)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 17
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        U, V, W, success = try_rank_als(17, seed=seed, max_iters=5000)
        if success:
            U_list = [[float(U[i][j]) for j in range(17)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(17)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(17)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 18
    for seed in range(5):
        if time.time() > deadline - 15:
            break
        U, V, W, success = try_rank_als(18, seed=seed, max_iters=3000)
        if success:
            U_list = [[float(U[i][j]) for j in range(18)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(18)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(18)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
