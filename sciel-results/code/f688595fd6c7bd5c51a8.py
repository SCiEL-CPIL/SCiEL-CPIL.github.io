"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    rank_max = n * m * p  # 30

    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 110s margin within 120s

    best_rank = rank_max
    best_result = None

    # Precompute the tensor structure
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want U, V, W such that for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    # Start from rank 13 (known lower bound for some small cases is around n*m*p/2)
    # Actually for 2x3x5, the trivial is 30. Let's try from 20 down.
    # Known results: matrix multiplication 2x3x5 has rank at most 24 (from literature)
    # Let's try ranks from 24 down to 15

    # Build the linear system for a given rank r
    # Variables: U has n*m*r vars, V has m*p*r vars, W has p*n*r vars
    # Total: (n*m + m*p + p*n) * r = (6 + 15 + 10) * r = 31*r variables
    # Constraints: n*m*p = 30 equations, each is cubic (degree 3)
    # This is a system of 30 polynomial equations in 31*r variables.
    
    # For small r, we can try to solve using a grid search or random initialization
    # followed by gradient descent / Newton's method.
    
    # Alternative: use the fact that we can fix one factor and solve for others.
    # For a fixed W, the problem becomes bilinear in U and V.
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let D[k*n+i, r] = W[k*n+i, r]
    # Then sum_r U[i*m+j, r] * (V[j*p+k, r] * D[k*n+i, r]) = 1
    # This is still coupled.
    
    # Let me try a different approach: random search with refinement.
    # Initialize U, V, W randomly with small values, then do alternating minimization.
    
    def check_exact(U, V, W, r):
        """Check if the decomposition is exact with entries in {0, 0.5, 1, 1.5, ...}"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def check_half_integer(U, V, W, r):
        """Check if all entries are multiples of 0.5"""
        for matrix in [U, V, W]:
            for row in matrix:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-6:
                        return False
        return True

    def try_rank(r, max_iter=200):
        """Try to find a rank-r decomposition using alternating minimization."""
        if r >= rank_max:
            return None
        
        # Multiple random restarts
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Alternating minimization
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update W: for each (k, i), solve for W[k*n+i, :]
                # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # Let A_{j, r} = U[i*m+j, r] * V[j*p+k, r]
                # Then A @ W[k*n+i, :] = 1_vector
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for rr in range(r):
                                A[j, rr] = U[i*m+j, rr] * V[j*p+k, rr]
                        # Solve A @ x = ones(m)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Update U: for each (i, j), solve for U[i*m+j, :]
                # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for rr in range(r):
                                A[k, rr] = V[j*p+k, rr] * W[k*n+i, rr]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j, k), solve for V[j*p+k, :]
                # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for rr in range(r):
                                A[i, rr] = U[i*m+j, rr] * W[k*n+i, rr]
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(r):
                                val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            err += (val - 1.0) ** 2
                if err < 1e-8:
                    # Check if entries are close to half-integers
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    # Verify the rounded version
                    err_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for rr in range(r):
                                    val += U_rounded[i*m+j, rr] * V_rounded[j*p+k, rr] * W_rounded[k*n+i, rr]
                                err_rounded += (val - 1.0) ** 2
                    
                    if err_rounded < 1e-6:
                        # Return the solution
                        U_out = [[float(x) for x in row] for row in U_rounded]
                        V_out = [[float(x) for x in row] for row in V_rounded]
                        W_out = [[float(x) for x in row] for row in W_rounded]
                        return {"U": U_out, "V": V_out, "W": W_out}
                    
                    # If rounding doesn't work, try returning unrounded if exact enough
                    if err < 1e-10:
                        U_out = [[float(x) for x in row] for row in U]
                        V_out = [[float(x) for x in row] for row in V]
                        W_out = [[float(x) for x in row] for row in W]
                        return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    # Try ranks from 24 down to 15
    for r in range(24, 14, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=100)
        if result is not None:
            best_rank = r
            best_result = result
            # Found one, try lower ranks if time permits
            if r > 15:
                continue
            else:
                break
    
    # If no better rank found, use the schoolbook solution
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
