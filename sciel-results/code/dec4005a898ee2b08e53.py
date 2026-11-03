import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Lower bound
    min_rank = max(n * m, m * p, p * n)
    
    # Pre-compute index arrays for speed
    # For U: row index i*m+j
    # For V: row index j*p+k  
    # For W: row index k*n+i
    
    # Try ranks from min_rank upward
    max_rank_offset = 15
    base_rank = min_rank
    
    # Structured initialization patterns
    def structured_init_U(R, pattern=0):
        """Create structured initialization for U"""
        U = np.zeros((n * m, R))
        if pattern == 0:
            # Identity-like pattern
            for r in range(R):
                if r < n * m:
                    U[r, r] = 1.0
        elif pattern == 1:
            # Block structure
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    for r in range(R):
                        U[idx, r] = 1.0 / np.sqrt(R)
        elif pattern == 2:
            # Random but with structure
            np.random.seed(123 + pattern)
            U = np.random.randn(n * m, R) * 0.5
            # Add some structure
            for i in range(n):
                for j in range(m):
                    U[i * m + j, 0] += 1.0
        elif pattern == 3:
            # Kronecker-like
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    for r in range(R):
                        U[idx, r] = ((i + 1) * (j + 1)) / (R + 1)
        return U
    
    def structured_init_V(R, pattern=0):
        """Create structured initialization for V"""
        V = np.zeros((m * p, R))
        if pattern == 0:
            for r in range(R):
                if r < m * p:
                    V[r, r] = 1.0
        elif pattern == 1:
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    for r in range(R):
                        V[idx, r] = 1.0 / np.sqrt(R)
        elif pattern == 2:
            np.random.seed(124 + pattern)
            V = np.random.randn(m * p, R) * 0.5
            for j in range(m):
                for k in range(p):
                    V[j * p + k, 0] += 1.0
        elif pattern == 3:
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    for r in range(R):
                        V[idx, r] = ((j + 1) * (k + 1)) / (R + 1)
        return V
    
    def structured_init_W(R, pattern=0):
        """Create structured initialization for W"""
        W = np.zeros((p * n, R))
        if pattern == 0:
            for r in range(R):
                if r < p * n:
                    W[r, r] = 1.0
        elif pattern == 1:
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    for r in range(R):
                        W[idx, r] = 1.0 / np.sqrt(R)
        elif pattern == 2:
            np.random.seed(125 + pattern)
            W = np.random.randn(p * n, R) * 0.5
            for k in range(p):
                for i in range(n):
                    W[k * n + i, 0] += 1.0
        elif pattern == 3:
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    for r in range(R):
                        W[idx, r] = ((k + 1) * (i + 1)) / (R + 1)
        return W
    
    def compute_error(U, V, W):
        """Compute the tensor decomposition error"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def als_step(U, V, W, R, max_iters=200, tol=1e-8):
        """Perform ALS optimization"""
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Solve for U
            A_u = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    W_idx = [k * n + i for k in range(p)]
                    W_slice = W[W_idx, :]
                    V_slice = V[j * p:(j + 1) * p, :]
                    A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
            
            norms_sq = np.sum(A_u * A_u, axis=1)
            norms_sq[norms_sq < 1e-12] = 1e-12
            U_new = A_u / norms_sq[:, np.newaxis]
            
            # Solve for V
            A_v = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    U_idx = [i * m + j for i in range(n)]
                    U_slice = U_new[U_idx, :]
                    W_idx = [k * n + i for i in range(n)]
                    W_slice = W[W_idx, :]
                    A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
            
            norms_sq_v = np.sum(A_v * A_v, axis=1)
            norms_sq_v[norms_sq_v < 1e-12] = 1e-12
            V_new = A_v / norms_sq_v[:, np.newaxis]
            
            # Solve for W
            A_w = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    U_idx = [i * m + j for j in range(m)]
                    U_slice = U_new[U_idx, :]
                    V_idx = [j * p + k for j in range(m)]
                    V_slice = V_new[V_idx, :]
                    A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
            
            norms_sq_w = np.sum(A_w * A_w, axis=1)
            norms_sq_w[norms_sq_w < 1e-12] = 1e-12
            W_new = A_w / norms_sq_w[:, np.newaxis]
            
            # Calculate error
            err = compute_error(U_new, V_new, W_new)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            if best_error < tol:
                break
            
            U, V, W = U_new, V_new, W_new
        
        return best_U, best_V, best_W, best_error
    
    def try_rank(R, max_patterns=4, max_restarts=3):
        """Try to find a decomposition of rank R"""
        for pattern in range(max_patterns):
            if time.time() > deadline:
                return None
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    return None
                
                # Create initialization
                if pattern == 0:
                    U = structured_init_U(R, 0)
                    V = structured_init_V(R, 0)
                    W = structured_init_W(R, 0)
                elif pattern == 1:
                    U = structured_init_U(R, 1)
                    V = structured_init_V(R, 1)
                    W = structured_init_W(R, 1)
                elif pattern == 2:
                    U = structured_init_U(R, 2)
                    V = structured_init_V(R, 2)
                    W = structured_init_W(R, 2)
                else:
                    U = structured_init_U(R, 3)
                    V = structured_init_V(R, 3)
                    W = structured_init_W(R, 3)
                
                # Add small random perturbation for restarts > 0
                if restart > 0:
                    np.random.seed(42 + pattern * 100 + restart)
                    U = U + np.random.randn(*U.shape) * 0.1
                    V = V + np.random.randn(*V.shape) * 0.1
                    W = W + np.random.randn(*W.shape) * 0.1
                
                # Run ALS
                U_opt, V_opt, W_opt, err = als_step(U, V, W, R, max_iters=300, tol=1e-10)
                
                if err < 1e-6:
                    # Try rounding to half-integers
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    
                    is_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                                if abs(val - 1.0) > 1e-9:
                                    is_exact = False
                                    break
                            if not is_exact:
                                break
                        if not is_exact:
                            break
                    
                    if is_exact:
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        return None
    
    # Try ranks from lower bound upward
    for rank_offset in range(max_rank_offset):
        if time.time() > deadline:
            break
        
        R = base_rank + rank_offset
        result = try_rank(R)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
