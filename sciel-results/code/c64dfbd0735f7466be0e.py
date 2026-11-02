"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def try_structured_rank(r, max_trials=30):
        """Try to find a rank-r decomposition using structured patterns."""
        nonlocal best_U, best_V, best_W, best_rank
        
        # For 2x4 x 4x7, we can use a block structure
        # The 4x4 middle part can use Strassen-like methods
        # Let's try various structured approaches
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Strategy 1: Use Kronecker product structure
            # For 2x4 x 4x7, think of 4x4 as 2x2 blocks
            # Each 2x2 block multiplication has rank 7 (Strassen)
            # But we need to handle the 2 and 7 dimensions carefully
            
            # Try random sparse initialization with structured patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with sparse random values
            for d in range(r):
                # Each factor vector should have a specific sparsity pattern
                # Based on the structure of matrix multiplication
                u_idx = np.random.choice(nm, size=max(1, r//4), replace=False)
                v_idx = np.random.choice(mp, size=max(1, r//4), replace=False)
                w_idx = np.random.choice(pn, size=max(1, r//4), replace=False)
                
                for idx in u_idx:
                    U[idx, d] = np.random.choice([-1, 0, 1])
                for idx in v_idx:
                    V[idx, d] = np.random.choice([-1, 0, 1])
                for idx in w_idx:
                    W[idx, d] = np.random.choice([-1, 0, 1])
            
            # Alternating minimization with structured updates
            prev_err = float('inf')
            for it in range(100):
                if time.time() > deadline:
                    return False
                    
                # Update W
                K_W = np.einsum('ad,bd,abc->dc', U, V, T)
                G_W = (U.T @ U) * (V.T @ V)
                try:
                    W_new = np.linalg.solve(G_W + 1e-6 * np.eye(r), K_W)
                except np.linalg.LinAlgError:
                    break
                
                # Update U
                K_U = np.einsum('bd,cd,abc->ad', V, W_new, T)
                G_U = (V.T @ V) * (W_new.T @ W_new)
                try:
                    U_new = np.linalg.solve(G_U + 1e-6 * np.eye(r), K_U)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                K_V = np.einsum('ad,cd,abc->bd', U_new, W_new, T)
                G_V = (U_new.T @ U_new) * (W_new.T @ W_new)
                try:
                    V_new = np.linalg.solve(G_V + 1e-6 * np.eye(r), K_V)
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                T_rec = tensor_from_factors(U_new, V_new, W_new)
                err = np.linalg.norm(T_rec - T)
                
                U, V, W = U_new, V_new, W_new
                
                if err < 1e-12:
                    break
                if abs(prev_err - err) < 1e-14 and it > 20:
                    break
                prev_err = err
            
            # Final check with high precision
            T_rec = tensor_from_factors(U, V, W)
            if np.allclose(T_rec, T, atol=1e-10, rtol=1e-10):
                # Try to find exact half-integer solution
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-10, rtol=1e-10):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    return True
        return False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = 28
    # Known results suggest rank around 50-54 for this size
    lower_bound = 48
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        success = try_structured_rank(r, max_trials=20)
        if success:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
