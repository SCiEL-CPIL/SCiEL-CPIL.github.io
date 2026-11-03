"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140  # safety margin
    
    # Target: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use iterative refinement: fix two matrices, solve for third
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Initialize
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            # Alternating minimization
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update W: for each (k,i), solve min ||V[j*p+k,:] * W[k*n+i,:] * U[i*m+j,:] - 1||
                # This is: for each (i,j,k), the contribution is U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:])
                # Fix U and V, update W.
                # For a given (k,i), W[k*n+i,:] appears in equations for all j:
                #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                # This is: (U[i*m+0:m, :] * V[0:p,:]) ... wait, let me think differently.
                
                # Actually, for fixed U and V, the equations for W are:
                # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Group by (k,i): for fixed (k,i), we have m equations (one per j):
                #   sum_r [U[i*m+j,r] * V[j*p+k,r]] * W[k*n+i,r] = 1
                # Let A_ki[j, r] = U[i*m+j,r] * V[j*p+k,r]
                # Then A_ki @ W[k*n+i,:] = ones(m)
                # Solve this least squares for W[k*n+i,:]
                
                for k in range(p):
                    for i in range(n):
                        # Build A matrix: (m, r)
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve A @ x = ones(m)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            W[k*n+i, :] = rng.randn(r)
                
                # Update U: for each (i,j), equations for all k:
                #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # Group by (i,j): for fixed (i,j), we have p equations (one per k):
                #   sum_r U[i*m+j,r] * [V[j*p+k,r] * W[k*n+i,r]] = 1
                # Let B_ij[k, r] = V[j*p+k,r] * W[k*n+i,r]
                # Then B_ij @ U[i*m+j,:] = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, r))
                        for k in range(p):
                            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            x, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            U[i*m+j, :] = rng.randn(r)
                
                # Update V: for each (j,k), equations for all i:
                #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                # Group by (j,k): for fixed (j,k), we have n equations (one per i):
                #   sum_r V[j*p+k,r] * [U[i*m+j,r] * W[k*n+i,r]] = 1
                # Let C_jk[i, r] = U[i*m+j,r] * W[k*n+i,r]
                # Then C_jk @ V[j*p+k,:] = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        C = np.zeros((n, r))
                        for i in range(n):
                            C[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x, residuals, rank_C, sv = np.linalg.lstsq(C, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            V[j*p+k, :] = rng.randn(r)
                
                # Check convergence
                if verify(U, V, W, tol=1e-6):
                    # Found a valid decomposition
                    return U, V, W
            
            # Check at the end
            if verify(U, V, W, tol=1e-6):
                return U, V, W
        
        return None
    
    # Try ranks from best_rank-1 downward
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, tol=1e-8):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                # Continue trying lower ranks
            else:
                break
    
    # Convert to lists with integer/half-integer entries if possible
    # For now, return as-is; the evaluator should handle floats
    # But let's try to round to half-integers if close
    
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

