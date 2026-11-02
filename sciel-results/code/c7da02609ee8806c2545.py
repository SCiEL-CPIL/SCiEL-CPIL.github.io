"""Bootstrap spawn: direct structural construction with rank reduction via linear dependencies."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix.
    
    Strategy: Start with schoolbook (rank 96), then try to find lower-rank decompositions
    by solving the bilinear system using randomized initialization and iterative refinement.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Start with schoolbook as fallback
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower-rank decompositions using a direct approach
    # For rank r, we need to find U (12xr), V (32xr), W (24xr) such that
    # for all (i,j,k): sum_c U[i*4+j,c] * V[j*8+k,c] * W[k*3+i,c] = 1
    
    # Approach: Fix U and W, solve for V (or vice versa)
    # This is a linear system in the entries of V
    
    def try_rank_direct(r, seed=42):
        """Try to find a rank-r decomposition by fixing two matrices and solving for the third."""
        if r >= best_rank:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(100):
            # Initialize U and W randomly
            U_mat = rng.randn(n * m, r)
            W_mat = rng.randn(p * n, r)
            
            # For each (i,j,k), we need: sum_c U[i*4+j,c] * V[j*8+k,c] * W[k*3+i,c] = 1
            # This is a linear equation in V[j*8+k,c]
            
            # Build the system: for each (i,j,k), we have one equation
            # Unknowns: V[j*8+k,c] for all j,k,c -> 32*r unknowns
            # Equations: 96 (one per (i,j,k))
            
            # Actually, for fixed (j,k), V[j*8+k,:] appears in equations for all (i,k') where k'=k
            # and all (i',j) where j'=j
            
            # Let's set up the linear system properly
            # For each (i,j,k): sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1
            # Let A_{(i,j,k), (j,k,c)} = U[i*m+j,c] * W[k*n+i,c]
            # Then sum_c A_{(i,j,k), (j,k,c)} * V[j*p+k,c] = 1
            
            # But V[j*p+k,c] appears in multiple equations (for different i)
            # So we have a coupled system
            
            # Let's flatten: let x be a vector of length 32*r containing all V entries
            # For equation (i,j,k): sum_c U[i*m+j,c] * W[k*n+i,c] * x[j*p*? + ...] = 1
            
            # Actually, V has shape (m*p, r) = (32, r)
            # V[j*p+k, c] is at index j*p+k in the first dimension, c in the second
            # So V[j*p+k, c] = x[(j*p+k)*r + c]
            
            # For equation (i,j,k): sum_c U[i*m+j,c] * W[k*n+i,c] * x[(j*p+k)*r + c] = 1
            
            # Build the matrix A of shape (96, 32*r)
            A = np.zeros((n*m*p, m*p*r))
            b = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            A[eq_idx, (j*p+k)*r + c] = U_mat[i*m+j, c] * W_mat[k*n+i, c]
                        eq_idx += 1
            
            # Solve A @ x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is good
            if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                V_mat = x.reshape(m*p, r)
                
                # Verify
                if verify(U_mat, V_mat, W_mat):
                    return (U_mat, V_mat, W_mat)
            
            # Also try fixing U and V, solving for W
            U_mat = rng.randn(n * m, r)
            V_mat = rng.randn(m * p, r)
            
            # For equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # W has shape (p*n, r) = (24, r)
            # W[k*n+i, c] = y[(k*n+i)*r + c]
            
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            A[eq_idx, (k*n+i)*r + c] = U_mat[i*m+j, c] * V_mat[j*p+k, c]
                        eq_idx += 1
            
            try:
                y, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                W_mat = y.reshape(p*n, r)
                
                if verify(U_mat, V_mat, W_mat):
                    return (U_mat, V_mat, W_mat)
        
        return None
    
    # Try ranks from 95 down to 85
    for r in range(95, 84, -1):
        if time.time() > deadline:
            break
        
        result = try_rank_direct(r)
        if result is not None:
            U_mat, V_mat, W_mat = result
            best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
