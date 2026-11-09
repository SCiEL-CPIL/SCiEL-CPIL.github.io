"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast numpy verification."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_int(mat):
    """Snap a numpy array to nearest half-integers (multiples of 0.5)."""
    return np.round(mat * 2) / 2.0

def _snap_to_int(mat):
    """Snap a numpy array to nearest integers."""
    return np.round(mat)

def _try_decomposition(r, n=2, m=4, p=6, max_iter=500, seed=42):
    """
    Try to find a rank-r decomposition using iterative refinement.
    Uses a least-squares approach on the tensor unfolding.
    """
    np.random.seed(seed)
    
    # Initialize randomly with small values
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # Scale to make the sum of products approximately 1
    norm = np.mean(np.abs(U) * np.abs(V) * np.abs(W))
    if norm > 0:
        scale = (1.0 / (norm * r)) ** (1.0/3)
        U *= scale
        V *= scale
        W *= scale
    
    # Iterative refinement: alternate fixing two factors and solving for the third
    for iteration in range(max_iter):
        # Solve for U given V, W
        # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is: U[i*m+j] @ (V[j*p+k] * W[k*n+i])^T ... not quite
        # Let's think of it as: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # Define M_{(i,j),k,c} = V[j*p+k,c] * W[k*n+i,c]
        # Then sum_c U[i*m+j,c] * M_{(i,j),k,c} = 1 for all k
        # This is a linear system: U[i*m+j] @ M_{(i,j)}^T = ones(p)
        # But M depends on k, so we need: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let A_{(i,j)} be a p x r matrix where A_{(i,j),k,c} = V[j*p+k,c] * W[k*n+i,c]
        # Then U[i*m+j] @ A_{(i,j)}^T = ones(p)
        # So U[i*m+j] = ones(p) @ (A_{(i,j)}^T A_{(i,j)})^{-1} A_{(i,j)}^T (least squares)
        
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j] @ A^T = ones(p)
                # i.e., A @ U[i*m+j]^T = ones(p)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-8:
            # Snap to half-integers and verify
            U_snapped = _snap_to_half_int(U)
            V_snapped = _snap_to_half_int(V)
            W_snapped = _snap_to_half_int(W)
            
            if _verify_np(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
            # Try integer snap
            U_int = _snap_to_int(U)
            V_int = _snap_to_int(V)
            W_int = _snap_to_int(W)
            if _verify_np(U_int, V_int, W_int, n, m, p, tol=1e-9):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _schoolbook(n=2, m=4, p=6):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try ranks from low to high
    # Known lower bounds suggest rank is at least around 20-30 for this problem
    # Try a range of ranks
    for r in range(10, 49):
        if time.time() > deadline - 5:
            break
        
        result = _try_decomposition(r, n, m, p, max_iter=300, seed=r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = r
                U_fb, V_fb, W_fb = U, V, W
                break  # Found the minimal rank
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
