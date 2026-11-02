"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strassen coefficients for M_{2,2,2}
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook coefficients for M_{2,2,1}
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Build fallback: 7 Strassen + 1 schoolbook = rank 53
    def build_fallback():
        rank = 7 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
            col_offset += 7
        
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = U_school[i * m + j, r]
            for j in range(m):
                V[j * p + k_last, c] = V_school[j, r]
            for i in range(n):
                W[k_last * n + i, c] = W_school[i, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find M_{2,2,3} with rank 10
    def try_m223_rank10():
        """Try to find rank-10 decomposition for M_{2,2,3} using gradient-based optimization."""
        if time.time() > deadline - 10:
            return None
        
        # M_{2,2,3}: i in {0,1}, j in {0,1}, k in {0,1,2}
        # Tensor: T[i,j,k] = delta_{i,k}
        # We need U (4x10), V (6x10), W (6x10) such that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        # The tensor entries: T[i,j,k] = 1 if i==k else 0
        # So nonzero entries: (0,0,0)=1, (0,1,0)=1, (1,0,1)=1, (1,1,1)=1
        # Zero entries: (0,0,1)=0, (0,1,1)=0, (1,0,0)=0, (1,1,0)=0, (0,0,2)=0, (0,1,2)=0, (1,0,2)=0, (1,1,2)=0
        
        # Total: 4 nonzero + 8 zero = 12 constraints
        # Unknowns: 4*10 + 6*10 + 6*10 = 100
        # Very underdetermined - should be solvable
        
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U = np.random.randn(4, 10) * 0.1
            V = np.random.randn(6, 10) * 0.1
            W = np.random.randn(6, 10) * 0.1
            
            # Define the residual function
            def compute_residuals(U, V, W):
                residuals = np.zeros(12)
                idx = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            val = 0.0
                            for r in range(10):
                                val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                            target = 1.0 if i == k else 0.0
                            residuals[idx] = val - target
                            idx += 1
                return residuals
            
            # Use scipy's least_squares for nonlinear optimization
            from scipy.optimize import least_squares
            
            def flatten_params(x):
                U_flat = x[0:40].reshape(4, 10)
                V_flat = x[40:100].reshape(6, 10)
                W_flat = x[100:160].reshape(6, 10)
                return U_flat, V_flat, W_flat
            
            def objective(x):
                U_f, V_f, W_f = flatten_params(x)
                return compute_residuals(U_f, V_f, W_f)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=500)
                
                if result.success or np.max(np.abs(result.fun)) < 1e-8:
                    U_opt, V_opt, W_opt = flatten_params(result.x)
                    
                    # Verify the solution
                    res = compute_residuals(U_opt, V_opt, W_opt)
                    if np.max(np.abs(res)) < 1e-6:
                        # Round to nearest integer/half-integer if close
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        res_rounded = compute_residuals(U_rounded, V_rounded, W_rounded)
                        if np.max(np.abs(res_rounded)) < 1e-6:
                            return U_rounded, V_rounded, W_rounded
                        else:
                            return U_opt, V_opt, W_opt
            except Exception:
                continue
        
        return None
    
    # Try to find M_{2,2,3} with rank 10 to get total rank 52
    result = try_m223_rank10()
    
    if result is not None:
        # Build full solution with rank 52
        rank = 6 * 7 + 10
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # 6 Strassen blocks for k=0..11
        for block in range(6):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = block * 7 + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
        
        # M_{2,2,3} for k=12,13,14
        U_223, V_223, W_223 = result
        U[:, 42:] = U_223
        V[:, 42:] = V_223
        W[:, 42:] = W_223
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return build_fallback()
# EVOLVE-BLOCK-END
