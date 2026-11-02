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
        """Try to find rank-10 decomposition for M_{2,2,3} using improved optimization."""
        if time.time() > deadline - 10:
            return None
        
        # Precompute tensor targets
        # T[i,j,k] = 1 if i==k else 0
        # Indices: i in {0,1}, j in {0,1}, k in {0,1,2}
        # Total: 2*2*3 = 12 entries
        
        from scipy.optimize import least_squares
        
        def compute_residuals(U, V, W):
            """Compute residuals for M_{2,2,3} decomposition."""
            residuals = np.zeros(12)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.dot(
                            np.dot(U[i*2+j, :], V[j*3+k, :]),
                            W[k*2+i, :]
                        )
                        target = 1.0 if i == k else 0.0
                        residuals[idx] = val - target
                        idx += 1
            return residuals
        
        def flatten_params(x):
            U_flat = x[0:40].reshape(4, 10)
            V_flat = x[40:100].reshape(6, 10)
            W_flat = x[100:160].reshape(6, 10)
            return U_flat, V_flat, W_flat
        
        def objective(x):
            U_f, V_f, W_f = flatten_params(x)
            return compute_residuals(U_f, V_f, W_f)
        
        # Try multiple random initializations with different scales
        np.random.seed(42)
        
        best_solution = None
        best_residual = np.inf
        
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Use various initialization strategies
            if attempt < 10:
                # Small random initialization
                scale = 0.1
                U = np.random.randn(4, 10) * scale
                V = np.random.randn(6, 10) * scale
                W = np.random.randn(6, 10) * scale
            elif attempt < 20:
                # Medium random initialization
                scale = 0.5
                U = np.random.randn(4, 10) * scale
                V = np.random.randn(6, 10) * scale
                W = np.random.randn(6, 10) * scale
            else:
                # Structured initialization: start with partial solution
                # Use identity-like structure for first few columns
                U = np.zeros((4, 10))
                V = np.zeros((6, 10))
                W = np.zeros((6, 10))
                
                # Column 0: handle (i=0,j=0,k=0)
                U[0, 0] = 1.0
                V[0, 0] = 1.0
                W[0, 0] = 1.0
                
                # Column 1: handle (i=0,j=1,k=0)
                U[1, 1] = 1.0
                V[3, 1] = 1.0
                W[0, 1] = 1.0
                
                # Column 2: handle (i=1,j=0,k=1)
                U[2, 2] = 1.0
                V[1, 2] = 1.0
                W[2, 2] = 1.0
                
                # Column 3: handle (i=1,j=1,k=1)
                U[3, 3] = 1.0
                V[4, 3] = 1.0
                W[3, 3] = 1.0
                
                # Add small random perturbation to remaining columns
                U[:, 4:] = np.random.randn(4, 6) * 0.1
                V[:, 4:] = np.random.randn(6, 6) * 0.1
                W[:, 4:] = np.random.randn(6, 6) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use Levenberg-Marquardt with tight tolerance
                result = least_squares(
                    objective, x0, method='lm', max_nfev=1000,
                    ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
                
                U_opt, V_opt, W_opt = flatten_params(result.x)
                res = compute_residuals(U_opt, V_opt, W_opt)
                max_res = np.max(np.abs(res))
                
                if max_res < best_residual:
                    best_residual = max_res
                    best_solution = (U_opt, V_opt, W_opt)
                
                # If we have a good solution, try to polish it
                if max_res < 1e-8:
                    # Try rounding to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    res_rounded = compute_residuals(U_rounded, V_rounded, W_rounded)
                    if np.max(np.abs(res_rounded)) < 1e-6:
                        return U_rounded, V_rounded, W_rounded
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    res_int = compute_residuals(U_int, V_int, W_int)
                    if np.max(np.abs(res_int)) < 1e-6:
                        return U_int, V_int, W_int
                    
                    # Return the unrounded solution if it's good enough
                    if max_res < 1e-10:
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                continue
        
        # If we found a good solution but couldn't round it, return it
        if best_solution is not None and best_residual < 1e-6:
            return best_solution
        
        return None
    
    # Try to find M_{2,2,3} with rank 10 to get total rank 52
    result = try_m223_rank10()
    
    if result is not None:
        # Final verification of the complete solution
        U_test, V_test, W_test = result
        # Verify M_{2,2,3} part
        def verify_223(U, V, W):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.dot(np.dot(U[i*2+j, :], V[j*3+k, :]), W[k*2+i, :])
                        target = 1.0 if i == k else 0.0
                        if abs(val - target) > 1e-6:
                            return False
            return True
        
        if not verify_223(U_test, V_test, W_test):
            result = None
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
