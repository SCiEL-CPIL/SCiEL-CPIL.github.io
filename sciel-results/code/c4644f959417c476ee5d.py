"""Rewritten program with optimized tensor decomposition search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if (U,V,W) is a valid decomposition."""
        n, m, p = DIMENSIONS
        # Residual calculation
        # Target: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if j==j' (matched indices)
        # Actually, the condition is:
        # For all i, k, j: coeff of A[i,j] B[j,k] in C[i,k] is 1.
        # For all i, k, j, j' with j!=j': coeff of A[i,j] B[j',k] in C[i,k] is 0.
        
        # Vectorized check
        # U shape: (n*m, R)
        # V shape: (m*p, R)
        # W shape: (p*n, R)
        
        # We need to check for all i, k, j, j'
        # This is equivalent to checking the tensor T_{i j j' k}
        
        # Efficient check:
        # Let M_U be matrix of shape (n, m, R)
        # Let M_V be matrix of shape (m, p, R)
        # Let M_W be matrix of shape (p, n, R)
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # We want to verify:
        # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        # Sum_r U[i,j,r] V[j',k,r] W[k,i,r] == 0 for j != j'
        
        # Let's compute the full tensor C[i, k, j, j'] = Sum_r U[i,j,r] V[j',k,r] W[k,i,r]
        # This is a 4D tensor.
        # We can compute it via einsum:
        # C[i, k, j, j'] = einsum('ijr,jkr,kir->ijkj', U_3d, V_3d, W_3d) -- wait indices
        # U is (n, m, R) -> i, j, r
        # V is (m, p, R) -> j', k, r  (Note: V index is j' for row, k for col)
        # W is (p, n, R) -> k, i, r
        
        # We want to sum over r.
        # Result index: i, j, j', k
        # C[i, j, j', k] = Sum_r U[i, j, r] * V[j', k, r] * W[k, i, r]
        
        # Let's use np.einsum
        # 'ijr,jkr,kir->ijjk'
        # U: i, j, r
        # V: j', k, r  (let's call the first index of V 'jp' to avoid confusion)
        # W: k, i, r
        
        # C[i, j, jp, k] = Sum_r U[i, j, r] * V[jp, k, r] * W[k, i, r]
        
        # We check:
        # If j == jp, C[i, j, j, k] should be 1.
        # If j != jp, C[i, j, jp, k] should be 0.
        
        try:
            C = np.einsum('ijr,jkr,kir->ijjk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            return False
            
        # Extract diagonal where j == jp
        # C_diag[i, j, k] = C[i, j, j, k]
        # Indices: i (0), j (1), jp (2), k (3)
        # We need C[i, j, j, k]
        # np.einsum('ijjk->ijk', C) extracts the diagonal over indices 1 and 2.
        
        C_diag = np.einsum('ijjk->ijk', C)
        
        # Check C_diag == 1
        if not np.allclose(C_diag, 1.0, atol=tol):
            return False
            
        # Check off-diagonal == 0
        # We can compute the sum of squares of all elements and subtract the diagonal squares
        # Sum_all = np.sum(C**2)
        # Sum_diag = np.sum(C_diag**2)
        # Off_diag_sum = Sum_all - Sum_diag
        # If Off_diag_sum > tol**2 * count, fail
        
        sum_all = np.sum(C**2)
        sum_diag = np.sum(C_diag**2)
        off_diag_sq_sum = sum_all - sum_diag
        
        # The number of off-diagonal elements is n*m*m*p - n*m*p
        # If off_diag_sq_sum is small, it's valid
        if off_diag_sq_sum > (tol**2) * (n * m * m * p):
            return False
            
        return True

    def residual(U, V, W):
        """Compute the residual (how far from valid decomposition)."""
        n, m, p = DIMENSIONS
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        try:
            C = np.einsum('ijr,jkr,kir->ijjk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            return 1e10
            
        C_diag = np.einsum('ijjk->ijk', C)
        
        # Error 1: C_diag should be 1
        err1 = np.sum((C_diag - 1.0)**2)
        
        # Error 2: Off-diagonals should be 0
        # We can just sum the squares of the whole tensor C, but we need to subtract the diagonal part?
        # No, the target is 0 for off-diagonals.
        # So we want Sum(C[i,j,jp,k]^2) to be 0 for j!=jp.
        # And Sum(C[i,j,j,k]^2) to be 1.
        
        # Total residual = Sum_{i,j,jp,k} (C[i,j,jp,k] - Target[i,j,jp,k])^2
        # Target is 1 if j==jp, 0 otherwise.
        
        # We can compute this efficiently:
        # Sum(C^2) - 2 * Sum(C * Target) + Sum(Target^2)
        # Sum(Target^2) = n*m*p (since there are n*m*p ones)
        # Sum(C * Target) = Sum(C_diag * 1) = Sum(C_diag)
        
        sum_c_sq = np.sum(C**2)
        sum_c_diag = np.sum(C_diag)
        
        residual_val = sum_c_sq - 2.0 * sum_c_diag + (n * m * p)
        
        return residual_val

    # Search for lower ranks
    # Start from schoolbook rank - 1 down to a reasonable limit
    # Given time constraints, we try a few ranks.
    
    # Lower bound is max(n*m, m*p, p*n) = 28.
    # But realistically, we can't go much lower than 84 for these dims without specific algos.
    # We will try to find if 83, 82, ... are possible.
    
    for target_rank in range(rank_sb - 1, max(28, rank_sb - 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations for this rank
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize with schoolbook columns + noise
            U_init = U_sb[:, :target_rank].copy()
            V_init = V_sb[:, :target_rank].copy()
            W_init = W_sb[:, :target_rank].copy()
            
            # Add noise to escape local minima
            if attempt > 0:
                noise = 0.5 * attempt
                U_init += noise * np.random.randn(n * m, target_rank)
                V_init += noise * np.random.randn(m * p, target_rank)
                W_init += noise * np.random.randn(p * n, target_rank)
            
            # Flatten for optimization
            U_flat = U_init.flatten()
            V_flat = V_init.flatten()
            W_flat = W_init.flatten()
            
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            def objective(x):
                if time.time() > deadline - 20:
                    return 1e10
                U = x[:n*m*target_rank].reshape(n*m, target_rank)
                V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                return residual(U, V, W)
            
            try:
                # Use L-BFGS-B for speed
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
                
                if res.fun < 1e-4:
                    # Verify with high precision
                    U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    if check_decomposition(U_sol, V_sol, W_sol, tol=1e-5):
                        best_rank = target_rank
                        best_U = U_sol.copy()
                        best_V = V_sol.copy()
                        best_W = W_sol.copy()
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
