import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i', j==j', k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, T, tol=1e-9):
        """Check if sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c]"""
        r = U.shape[1]
        D = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(D, T, atol=tol)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find a lower rank decomposition
    # Strategy: Use the structure of the problem
    # For each middle index j, we have a 3x4 sub-problem
    # Try ranks from 28 to 35 (Lader proved 33 is achievable for 3x3x4)
    
    best_result = None
    best_rank = n * m * p
    
    # First, try a structured approach based on known constructions
    # For 3x3x3, there are rank-21 decompositions
    # For 3x3x4, we can try to extend
    
    # Approach: Try to find a decomposition by solving linear equations
    # for a given rank, using random initialization and iterative refinement
    
    def try_rank(r, max_iter=200, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        rng = np.random.default_rng(seed)
        
        # Initialize with schoolbook and randomly merge some terms
        U_sb, V_sb, W_sb = schoolbook()
        
        # Start with schoolbook and try to reduce rank by finding linear dependencies
        # or by random perturbation and projection
        
        # Alternative: random initialization with structured values
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for attempt in range(50):
            if time.time() - start_time > time_limit - 5:
                break
                
            U = rng.choice(values, size=(n*m, r))
            V = rng.choice(values, size=(m*p, r))
            W = rng.choice(values, size=(p*n, r))
            
            # Check if valid
            if verify(U, V, W, T):
                return U, V, W
            
            # Iterative refinement: try to fix one entry at a time
            # This is a heuristic approach
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Compute error
                D = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - D
                
                if np.allclose(E, 0, atol=1e-9):
                    return U, V, W
                
                # Find the entry with largest error
                # and try to adjust the corresponding factor
                # This is complex, so we do random perturbation instead
                
                # Randomly perturb one entry
                choice = rng.integers(3)  # 0=U, 1=V, 2=W
                if choice == 0:
                    idx = rng.integers(n*m)
                    col = rng.integers(r)
                    old_val = U[idx, col]
                    U[idx, col] = rng.choice(values)
                    if not verify(U, V, W, T):
                        # Check if error decreased
                        D_new = np.einsum('at,bt,ct->abc', U, V, W)
                        E_new = T - D_new
                        if np.sum(E_new**2) < np.sum(E**2) - 1e-12:
                            E = E_new
                        else:
                            U[idx, col] = old_val
                elif choice == 1:
                    idx = rng.integers(m*p)
                    col = rng.integers(r)
                    old_val = V[idx, col]
                    V[idx, col] = rng.choice(values)
                    if not verify(U, V, W, T):
                        D_new = np.einsum('at,bt,ct->abc', U, V, W)
                        E_new = T - D_new
                        if np.sum(E_new**2) < np.sum(E**2) - 1e-12:
                            E = E_new
                        else:
                            V[idx, col] = old_val
                else:
                    idx = rng.integers(p*n)
                    col = rng.integers(r)
                    old_val = W[idx, col]
                    W[idx, col] = rng.choice(values)
                    if not verify(U, V, W, T):
                        D_new = np.einsum('at,bt,ct->abc', U, V, W)
                        E_new = T - D_new
                        if np.sum(E_new**2) < np.sum(E**2) - 1e-12:
                            E = E_new
                        else:
                            W[idx, col] = old_val
        
        return None
    
    # Try ranks from 28 to 35
    for r in range(28, 36):
        if time.time() - start_time > time_limit - 10:
            break
        
        result = try_rank(r, max_iter=100, seed=r * 1000 + 42)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
