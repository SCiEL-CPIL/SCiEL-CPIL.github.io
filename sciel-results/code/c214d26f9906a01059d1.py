"""Matrix multiplication 3x5x7 - rank minimization via structured search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    
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
    
    def try_rank(r, num_attempts=30):
        """Try to find a decomposition of rank r using random V,W and solving for U."""
        if time.time() > deadline:
            return None
        
        # Use different seeds for diversity
        for seed_offset in range(10):
            if time.time() > deadline:
                return None
            
            random.seed(12345 + r * 100 + seed_offset)
            
            for attempt in range(num_attempts // 10 + 1):
                if time.time() > deadline:
                    return None
                
                # Generate random V and W with small integer entries
                # Use values from {-2, -1, 0, 1, 2}
                V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(m*p)]
                W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(p*n)]
                
                # For each i, solve for U[i*m+j, :] for j=0..m-1
                # The system: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                
                U = [[0]*r for _ in range(n*m)]
                
                valid = True
                for i in range(n):
                    # Build the system for this i
                    # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
                    # Equations: for j in range(m), k in range(p):
                    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    
                    A_mat = np.zeros((m*p, m*r))
                    b_vec = np.ones(m*p)
                    
                    for j in range(m):
                        for k in range(p):
                            eq_idx = j*p + k
                            for c in range(r):
                                A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                    
                    # Solve least squares
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integer/half-integer
                    sol_rounded = np.round(sol * 2) / 2
                    if np.max(np.abs(sol - sol_rounded)) > 1e-8:
                        # Try to find nearby half-integers
                        valid = False
                        break
                    
                    for j in range(m):
                        for c in range(r):
                            U[i*m+j][c] = float(sol_rounded[j*r + c])
                
                if not valid:
                    continue
                
                # Verify the full solution
                if _verify_np(U, V, W, n, m, p):
                    # Convert to exact fractions
                    U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                    V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                    W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                    
                    # Double check with fractions
                    if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                        return {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }
        
        return None
    
    # Fallback to schoolbook immediately if we don't find a better rank quickly.
    # Given the difficulty of finding a lower rank for 3x5x7, we prioritize robustness.
    # We will try a very limited search for ranks 100-104.
    
    for r in range(104, 99, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, num_attempts=10)
        if result is not None:
            return result
    
    return schoolbook()

# EVOLVE-BLOCK-END
