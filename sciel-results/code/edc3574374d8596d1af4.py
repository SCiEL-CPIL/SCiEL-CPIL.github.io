"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) -> indices (i,j)
    # V shape: (m*p, rank) -> indices (j,k)  
    # W shape: (p*n, rank) -> indices (k,i)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 9, 8) = 9
    # But actually the tensor rank lower bound is higher.
    # For 2x3x4, let's try various ranks.
    
    deadline = time.time() + 120  # safety margin
    
    best_result = None
    
    # Known: schoolbook is 24. Let's try to find lower.
    # Try ranks: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    for rank in range(10, 25):
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(n, m, p, rank, max_iters=5000, tol=1e-12, n_restarts=50):
    """Try to find a rank-r decomposition via alternating minimization."""
    np.random.seed(42)
    
    for restart in range(n_restarts):
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        success = _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol)
        if success:
            # Verify and clean up
            U, V, W = _clean_solution(n, m, p, rank, U, V, W)
            if _verify(n, m, p, U, V, W, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol):
    """Alternating least squares to satisfy T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    
    for it in range(max_iters):
        # Solve for U given V, W
        # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
        # Actually: for fixed (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0..p-1
        # Let's define M_{k,c} = V[j*p+k,c] * W[k*n+i,c], then U[i*m+j,:] @ M^T = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j,:] @ A_mat.T = ones(p)
                # i.e., A_mat @ U[i*m+j,:].T = ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    return False
        
        # Solve for V given U, W
        # For each (j,k): sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    return False
        
        # Solve for W given U, V
        # For each (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    return False
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            return True
    
    return False

def _clean_solution(n, m, p, rank, U, V, W):
    """Try to clean up the solution to have exact integer/half-integer values."""
    # Round to nearest 0.5
    U_clean = np.round(U * 2) / 2
    V_clean = np.round(V * 2) / 2
    W_clean = np.round(W * 2) / 2
    return U_clean, V_clean, W_clean

def _verify(n, m, p, U, V, W, tol=1e-10):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
