import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Validate that U, V, W form an exact decomposition."""
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute approximation
    T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
    
    return np.allclose(T_approx, T, atol=tol)

def _try_rank_structured(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using structured approach."""
    # For matrix multiplication, we can use a structured decomposition
    # where each term corresponds to a "slice" of the computation
    
    # Strategy: try different factorizations
    # For 3x3 * 3x8, we can think of it as computing 3x8 output matrix
    # Each output element C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try using Kronecker-like structure
    # Initialize with zeros and try to fill in structure
    
    np.random.seed(12345 + r)
    
    # Try multiple random initializations with structured patterns
    for attempt in range(5):
        if time.time() > deadline:
            return None
        
        # Pattern 1: Block diagonal-like
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Assign each term to a (i,j,k) combination
        # With r terms, we need to cover all n*m*p = 72 combinations
        # So r must be at least 72 for exact schoolbook
        # But we can try to share terms if possible
        
        if r < n * m * p:
            # Try to find a non-trivial decomposition
            # Use random but structured initialization
            for col in range(r):
                # Randomly assign which (i,j,k) this term helps with
                # But ensure coverage
                i = col % n
                j = (col // n) % m
                k = (col // (n*m)) % p
                if col < n*m*p:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
            
            # Check if this simple assignment works
            if _validate(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _optimize_rank(n, m, p, r, deadline, seed=42):
    """Try to find rank-r decomposition using optimization."""
    from scipy.optimize import minimize
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Random initialization
    np.random.seed(seed)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    def objective(params):
        if time.time() > deadline:
            return np.inf
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r]
        
        U_curr = U_flat.reshape(n*m, r)
        V_curr = V_flat.reshape(m*p, r)
        W_curr = W_flat.reshape(p*n, r)
        
        T_approx = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
        diff = T - T_approx
        return np.sum(diff**2)
        
    def gradient(params):
        if time.time() > deadline:
            return np.zeros_like(params)
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r]
        
        U_curr = U_flat.reshape(n*m, r)
        V_curr = V_flat.reshape(m*p, r)
        W_curr = W_flat.reshape(p*n, r)
        
        T_approx = np.einsum('ac,bc,dc->abd', U_curr, V_curr, W_curr)
        diff = T - T_approx
        
        grad_U = -2 * np.einsum('abd,bc,dc->ac', diff, V_curr, W_curr)
        grad_V = -2 * np.einsum('abd,ac,dc->bc', diff, U_curr, W_curr)
        grad_W = -2 * np.einsum('abd,ac,bc->dc', diff, U_curr, V_curr)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
    params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
    except Exception:
        return None
    
    if res.fun < 1e-8:
        params_opt = res.x
        U_opt = params_opt[:n*m*r].reshape(n*m, r)
        V_opt = params_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = params_opt[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r].reshape(p*n, r)
        
        # Round to half-integers
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        if _validate(U_round, V_round, W_round, n, m, p, tol=1e-9):
            return U_round, V_round, W_round
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try lower ranks in order
    # For 3x3 * 3x8, theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But practical lower bounds are higher. Try ranks from 60 down to 40
    ranks_to_try = list(range(70, 35, -5))  # 70, 65, 60, ..., 40
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # First try structured approach (fast)
        result = _try_rank_structured(n, m, p, r, deadline)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            continue
        
        # Then try optimization (slower)
        result = _optimize_rank(n, m, p, r, deadline, seed=42 + r)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            # If we found a good rank, try to go lower
            continue
    
    # Final validation
    if not _validate(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = _build_schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
