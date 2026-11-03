"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W, tol=1e-8):
        """Check if U,V,W give exact decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook fallback (rank 54)
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = None, None, None
    
    # Alternating least squares for fixed rank
    def als_solve(r, max_iters=200, tol=1e-10, seed=0):
        """Alternating minimization: fix W, solve for (U,V) via ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        W = rng.randn(p*n, r) * 0.5
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Fix W, solve for U (linear in U for fixed V,W)
            # Constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed V,W: this is linear in U
            # Build A_U * U_vec = b where U_vec is (n*m*r,)
            # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
            
            # Solve for U: A_U * U_flat = 1
            A_U = np.zeros((n*m*p, n*m*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i*m*p + j*p + k
                        for c in range(r):
                            A_U[row, (i*m+j)*r + c] = V[j*p+k, c] * W[k*n+i, c]
            
            b_U = np.ones(n*m*p)
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_flat.reshape(n*m, r)
            
            # Solve for V: A_V * V_flat = 1
            A_V = np.zeros((n*m*p, m*p*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i*m*p + j*p + k
                        for c in range(r):
                            A_V[row, (j*p+k)*r + c] = U[i*m+j, c] * W[k*n+i, c]
            
            b_V = np.ones(n*m*p)
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_flat.reshape(m*p, r)
            
            # Update W: A_W * W_flat = 1
            A_W = np.zeros((n*m*p, p*n*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i*m*p + j*p + k
                        for c in range(r):
                            A_W[row, (k*n+i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
            
            b_W = np.ones(n*m*p)
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_flat.reshape(p*n, r)
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err = max(err, abs(val - 1.0))
            
            if err < tol:
                return U, V, W, err
        
        # Final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    err = max(err, abs(val - 1.0))
        
        return U, V, W, err
    
    # Try ranks from low to high
    for r in range(6, min(schoolbook_rank + 1, 55)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            U, V, W, err = als_solve(r, max_iters=150, tol=1e-10, seed=seed)
            
            if err < 1e-8:
                # Verify with stricter tolerance
                if verify(U, V, W, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    # Return best found
    if best_U is not None:
        # Convert to integer/half-integer if possible
        def to_half_int(arr):
            result = np.round(arr * 2) / 2.0
            # Check if rounding to half-integers preserves validity
            if verify(result, 
                      np.round(V * 2) / 2.0 if best_V is not None else result,
                      np.round(W * 2) / 2.0 if best_W is not None else result, tol=1e-6):
                return result.tolist()
            return arr.tolist()
        
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
        # Try to snap to half-integers
        U_h = np.round(best_U * 2) / 2.0
        V_h = np.round(best_V * 2) / 2.0
        W_h = np.round(best_W * 2) / 2.0
        
        if verify(U_h, V_h, W_h, tol=1e-6):
            U_out = U_h.tolist()
            V_out = V_h.tolist()
            W_out = W_h.tolist()
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
