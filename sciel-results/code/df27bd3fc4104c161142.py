"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Searches for a rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
    Falls back to the trivial rank-63 decomposition if no lower rank is found.
    """
    n, m, p = 3, 3, 7
    deadline = time.time() + 120  # 120s wall-clock limit
    
    # Trivial rank-63 decomposition as fallback
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try to find lower rank decompositions
    # The constraint: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We search for integer/half-integer solutions
    
    def check_decomposition(U, V, W):
        """Check if U, V, W satisfy the decomposition constraint."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for s in range(r):
                        val += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(2*x) / 2.0
    
    def search_for_rank(r, max_iterations=1000):
        """Try to find a rank-r decomposition using random restarts and local search."""
        if time.time() > deadline - 5:
            return None
        
        for attempt in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Iterative optimization: alternate fixing two matrices and solving for the third
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j): sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all k
                # This is a linear system in U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_s U[i*m+j,s] * (V[j*p+k,s] * W[k*n+i,s]) = 1
                        A_sys = np.zeros((p, r))
                        b_sys = np.ones(p)
                        for k in range(p):
                            for s in range(r):
                                A_sys[k, s] = V[j*p+k, s] * W[k*n+i, s]
                        # Solve least squares
                        if r >= p:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        else:
                            # Underdetermined, use pseudoinverse
                            sol = np.linalg.pinv(A_sys) @ b_sys
                        U[i*m+j, :] = sol
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n, r))
                        b_sys = np.ones(n)
                        for i in range(n):
                            for s in range(r):
                                A_sys[i, s] = U[i*m+j, s] * W[k*n+i, s]
                        if r >= n:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        else:
                            sol = np.linalg.pinv(A_sys) @ b_sys
                        V[j*p+k, :] = sol
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        A_sys = np.zeros((m, r))
                        b_sys = np.ones(m)
                        for j in range(m):
                            for s in range(r):
                                A_sys[j, s] = U[i*m+j, s] * V[j*p+k, s]
                        if r >= m:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        else:
                            sol = np.linalg.pinv(A_sys) @ b_sys
                        W[k*n+i, :] = sol
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(r))
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Found a solution, try to round to half-integers
                    U_rounded = np.array([[to_half_integer(x) for x in row] for row in U])
                    V_rounded = np.array([[to_half_integer(x) for x in row] for row in V])
                    W_rounded = np.array([[to_half_integer(x) for x in row] for row in W])
                    if check_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    # Also try without rounding
                    if check_decomposition(U, V, W):
                        # Check if values are close to half-integers
                        if all(abs(x - to_half_integer(x)) < 1e-6 for row in U for x in row) and \
                           all(abs(x - to_half_integer(x)) < 1e-6 for row in V for x in row) and \
                           all(abs(x - to_half_integer(x)) < 1e-6 for row in W for x in row):
                            return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 25 to 62 (skip trivial 63)
    for r in range(25, best_rank):
        if time.time() > deadline - 10:
            break
        result = search_for_rank(r, max_iterations=50)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found the minimum rank
    
    return best_result
# EVOLVE-BLOCK-END
