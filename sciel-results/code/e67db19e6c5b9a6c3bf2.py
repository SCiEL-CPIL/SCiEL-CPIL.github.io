"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A
    multiplied by the (j,k) entry of B contributes to the (i,k) entry of C.
    
    We seek rank-r factors U (nm x r), V (mp x r), W (pn x r) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For n=2, m=2, p=6: nm=4, mp=12, pn=12, tensor size 4*12*12 = 576 entries.
    The schoolbook rank is 24. We search for lower ranks.
    
    Strategy: Try ranks from a lower bound upward. For each rank, use random
    initialization with alternating least squares to try to find an exact solution.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # = 24
    
    # Lower bound: max(nm, mp, pn) = 12
    start_rank = max(nm, mp, pn)  # = 12
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    def _verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = abs(val - T_full[a, b, c])
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def _try_rank(rank, n_trials=20, seed=None):
        """Try to find a rank-r decomposition via random init + ALS."""
        if seed is not None:
            np.random.seed(seed)
        
        T_flat = T_full.reshape(nm, mp * pn)
        T_flat_V = T_full.transpose(1, 0, 2).reshape(mp, nm * pn)
        T_flat_W = T_full.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with various strategies
            strategy = trial % 3
            if strategy == 0:
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
            elif strategy == 1:
                # Sparse initialization
                U = np.random.randn(nm, rank) * (np.random.rand(nm, rank) < 0.5)
                V = np.random.randn(mp, rank) * (np.random.rand(mp, rank) < 0.5)
                W = np.random.randn(pn, rank) * (np.random.rand(pn, rank) < 0.5)
            else:
                # Binary-like initialization
                U = np.random.choice([-1, 1], size=(nm, rank)).astype(float)
                V = np.random.choice([-1, 1], size=(mp, rank)).astype(float)
                W = np.random.choice([-1, 1], size=(pn, rank)).astype(float)
            
            # Scale to avoid numerical issues
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Alternating least squares
            converged = False
            for iteration in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M_U = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M_U[b * pn + c, :] = V[b, :] * W[c, :]
                
                try:
                    A_U = M_U.T @ M_U
                    B_U = M_U.T @ T_flat.T
                    U = np.linalg.solve(A_U, B_U).T
                except np.linalg.LinAlgError:
                    U = U + 0.01 * np.random.randn(nm, rank)
                
                # Update V
                M_V = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        M_V[a * pn + c, :] = U[a, :] * W[c, :]
                
                try:
                    A_V = M_V.T @ M_V
                    B_V = M_V.T @ T_flat_V.T
                    V = np.linalg.solve(A_V, B_V).T
                except np.linalg.LinAlgError:
                    V = V + 0.01 * np.random.randn(mp, rank)
                
                # Update W
                M_W = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        M_W[a * mp + b, :] = U[a, :] * V[b, :]
                
                try:
                    A_W = M_W.T @ M_W
                    B_W = M_W.T @ T_flat_W.T
                    W = np.linalg.solve(A_W, B_W).T
                except np.linalg.LinAlgError:
                    W = W + 0.01 * np.random.randn(pn, rank)
                
                # Check convergence
                error = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T_full[a, b, c]) ** 2
                
                if error < 1e-12:
                    converged = True
                    break
            
            if converged:
                if _verify(U, V, W, tol=1e-9):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try ranks from lower to higher
    best_result = None
    for rank in range(start_rank, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n_trials=15, seed=rank * 1000)
        if result is not None:
            best_result = result
            break
    
    # If no lower rank found, return schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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
