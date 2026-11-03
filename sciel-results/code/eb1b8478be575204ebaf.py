import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # T[i,j,k] = 1 for all i,j,k
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    deadline = time.time() + 120
    
    # Try different ranks
    for R in range(10, 31):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        # Use a structured initialization
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Initialize with a simple pattern
        for r in range(R):
            # Spread the values
            pass
        
        # Iterative solver
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Solve: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve A @ U[i*m+j, :]^T = ones
                    if np.linalg.matrix_rank(A) > 0:
                        # Minimum norm solution
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Solve: sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                    A = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    if np.linalg.matrix_rank(A) > 0:
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Solve: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    if np.linalg.matrix_rank(A) > 0:
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error += abs(val - 1)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-9:
                # Found a solution! Now round to exact values
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded solution is exact
                error_rounded = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                            error_rounded += abs(val - 1)
                
                if error_rounded < 1e-9:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we didn't find an exact solution for this R, continue to next R
    
    # Fallback: schoolbook
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
