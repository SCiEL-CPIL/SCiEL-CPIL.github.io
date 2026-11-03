"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    # Output index: W[k*n + i, r]
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (9, rank) 
    # W shape: (p*n, rank) = (6, rank)
    
    deadline = time.time() + 140  # safety margin
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's try a structured approach.
        # Start from schoolbook (rank 18) and try to merge terms.
        
        # Schoolbook: rank 18, U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for c=i*m*p+j*p+k
        # Let's try rank 17 first.
        
        # Approach: use random initialization and gradient descent
        np.random.seed(42)
        
        best_solution = None
        best_residual = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Flatten parameters
            def flatten(params):
                U_f = params[:n*m*r].reshape(n*m, r)
                V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
                return U_f, V_f, W_f
            
            def residual(params):
                U_f, V_f, W_f = flatten(params)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :])
                            res.append(val - 1.0)
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=5000)
                res_val = result.cost
                if res_val < best_residual:
                    best_residual = res_val
                    best_solution = result.x
            except Exception:
                pass
        
        if best_solution is None:
            return None
        
        # Check if residual is small enough to suggest exact solution
        if best_residual < 1e-10:
            U_f, V_f, W_f = flatten(best_solution)
            # Try to find rational approximations
            U_rat = []
            V_rat = []
            W_rat = []
            for row in U_f:
                U_rat.append([Fraction(x).limit_denominator(1000) for x in row])
            for row in V_f:
                V_rat.append([Fraction(x).limit_denominator(1000) for x in row])
            for row in W_f:
                W_rat.append([Fraction(x).limit_denominator(1000) for x in row])
            
            # Verify exactness
            def verify(U_r, V_r, W_r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(r):
                                s += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                            if s != 1:
                                return False
                return True
            
            if verify(U_rat, V_rat, W_rat):
                return [U_rat, V_rat, W_rat]
        
        return None
    
    # Try ranks from 16 upward
    for r in range(16, 19):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists of numbers (int or half-int)
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook rank 18
    rank = n * m * p
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
