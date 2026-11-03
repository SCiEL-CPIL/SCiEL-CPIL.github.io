"""Find rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 4]

def _rationalize_matrix(M, max_denom=100):
    """Convert a float matrix to a list of lists of Fractions (integer or half-integer)."""
    result = []
    for row in M:
        r = []
        for val in row:
            if abs(val) < 1e-10:
                r.append(Fraction(0, 1))
            else:
                # Try to find a simple rational approximation
                best = Fraction(val).limit_denominator(max_denom)
                # Check if it's close
                if abs(float(best) - val) < 1e-8:
                    r.append(best)
                else:
                    # Try with larger denominator
                    best2 = Fraction(val).limit_denominator(max_denom * 10)
                    if abs(float(best2) - val) < 1e-8:
                        r.append(best2)
                    else:
                        r.append(Fraction(val).limit_denominator(max_denom * 100))
        result.append(r)
    return result

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, max_iter=500):
    """Try to find a decomposition of given rank r."""
    # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
    # Total variables: (n*m + m*p + p*n) * r
    nm, mp, pn = n*m, m*p, p*n
    total_vars = (nm + mp + pn) * r
    
    # We'll use an iterative approach: initialize randomly, then optimize
    # The equations are bilinear, so we use alternating least squares
    
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            return None
        
        # Random initialization
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Alternating least squares
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
            # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])^T = 1
            # Let's form the system: for fixed (i,j), we have p equations in r unknowns
            for i in range(n):
                for j in range(m):
                    # Build A (p x r): A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve A @ u = b in least squares
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = u_sol
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = v_sol
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w_sol
                    except:
                        pass
        
        # Check if we found a valid solution
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
            # Rationalize
            U_rat = _rationalize_matrix(U)
            V_rat = _rationalize_matrix(V)
            W_rat = _rationalize_matrix(W)
            
            # Verify with exact fractions
            if _verify(U_rat, V_rat, W_rat, n, m, p, tol=1e-10):
                return (U_rat, V_rat, W_rat)
            
            # Try with tighter rationalization
            U_rat2 = _rationalize_matrix(U, max_denom=1000)
            V_rat2 = _rationalize_matrix(V, max_denom=1000)
            W_rat2 = _rationalize_matrix(W, max_denom=1000)
            if _verify(U_rat2, V_rat2, W_rat2, n, m, p, tol=1e-10):
                return (U_rat2, V_rat2, W_rat2)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 12
    lower_bound = max(n*m, m*p, p*n)
    schoolbook = n * m * p
    
    for r in range(lower_bound, schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = schoolbook
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
