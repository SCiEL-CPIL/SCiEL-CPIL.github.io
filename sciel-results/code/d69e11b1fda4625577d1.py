import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def rationalize_matrix(M, tol=1e-6):
        """Try to find rational approximations with small denominators."""
        rows = M.shape[0]
        cols = M.shape[1]
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                found = False
                # Try denominators 1 to 20 for better coverage
                for d in range(1, 21):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    if abs(float(frac) - val) < tol:
                        result[i][j] = frac
                        found = True
                        break
                if not found:
                    return None
        return result

    def is_half_integer(frac):
        """Check if frac * 2 is an integer."""
        return (frac * 2).denominator == 1

    def verify_fast(U, V, W, T):
        """Fast verification using numpy einsum."""
        # Check if entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        
        rank = len(U[0])
        # Convert to numpy for fast computation
        U_np = np.array([[float(x) for x in row] for row in U])
        V_np = np.array([[float(x) for x in row] for row in V])
        W_np = np.array([[float(x) for x in row] for row in W])
        
        # Compute tensor product
        T_calc = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Compare with target
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def als_decompose(T, R, max_iters=30, tol=1e-8, seed=None):
        """ALS decomposition with random initialization."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        err = float('inf')
        for it in range(max_iters):
            # Update U
            for a in range(n*m):
                A = np.zeros((m*p * p*n, R))
                y = np.zeros(m*p * p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                
            # Update V
            for b in range(m*p):
                A = np.zeros((n*m * p*n, R))
                y = np.zeros(n*m * p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                
            # Update W
            for c in range(p*n):
                A = np.zeros((n*m * m*p, R))
                y = np.zeros(n*m * m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    def qr_rank_reveal(T, R, num_trials=3, seed=None):
        """Use QR decomposition with random projections to find low-rank structure."""
        if seed is not None:
            np.random.seed(seed)
        
        # Flatten T to a matrix for SVD/QR analysis
        # T shape: (n*m, m*p, p*n) = (9, 12, 12)
        # Flatten to (9, 12*12) = (9, 144)
        T_flat = T.reshape(n*m, m*p * p*n)
        
        # Try multiple random projections
        for trial in range(num_trials):
            # Random projection to reduce dimension
            proj_dim = min(20, m*p*p*n)
            P = np.random.randn(m*p*p*n, proj_dim)
            T_proj = T_flat @ P  # (n*m, proj_dim)
            
            # QR decomposition
            Q, R_mat = np.linalg.qr(T_proj)
            
            # Use the first R columns of Q as initial U
            if R <= proj_dim:
                U_init = Q[:, :R]
            else:
                U_init = Q[:, :min(R, proj_dim)]
                # Pad with random if needed
                if U_init.shape[1] < R:
                    pad = np.random.randn(n*m, R - U_init.shape[1])
                    U_init = np.hstack([U_init, pad])
            
            # Initialize V and W randomly
            V_init = np.random.randn(m*p, R)
            W_init = np.random.randn(p*n, R)
            
            # Run ALS from this initialization
            U, V, W, err = als_decompose(T, R, max_iters=20, tol=1e-8)
            if err < 1e-6:
                return U, V, W, err
        
        return None, None, None, float('inf')

    # Search for minimal R
    # Known lower bound for 3x3x4 is around 21
    # Try from 18 upward to find the minimum
    best_result = None
    
    # First try QR-based approach for lower ranks
    for R in range(18, 26):
        if time.time() > deadline - 15:
            break
        
        # Try QR-based initialization
        U, V, W, err = qr_rank_reveal(T, R, num_trials=2, seed=R)
        if err < 1e-6:
            U_frac = rationalize_matrix(U, tol=1e-5)
            V_frac = rationalize_matrix(V, tol=1e-5)
            W_frac = rationalize_matrix(W, tol=1e-5)
            if U_frac and V_frac and W_frac:
                if verify_fast(U_frac, V_frac, W_frac, T):
                    U_out = [[float(x) for x in row] for row in U_frac]
                    V_out = [[float(x) for x in row] for row in V_frac]
                    W_out = [[float(x) for x in row] for row in W_frac]
                    return {"U": U_out, "V": V_out, "W": W_out}
        
        # Try standard ALS with multiple restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            U, V, W, err = als_decompose(T, R, max_iters=25, seed=R * 100 + restart)
            if err < 1e-6:
                U_frac = rationalize_matrix(U, tol=1e-5)
                V_frac = rationalize_matrix(V, tol=1e-5)
                W_frac = rationalize_matrix(W, tol=1e-5)
                if U_frac and V_frac and W_frac:
                    if verify_fast(U_frac, V_frac, W_frac, T):
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    return schoolbook()
