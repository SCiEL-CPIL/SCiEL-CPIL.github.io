"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank 36
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly reproduces the matmul tensor."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        rank = U_arr.shape[1]
        
        # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Compare with standard matmul tensor: T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
        # The matmul tensor has T[i,j,k] = 1 for all (i,j,k) in the range, 
        # actually no - the tensor is defined as T[i,j,k] = 1 for the standard basis
        # Wait, the matmul tensor T satisfies: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k)
        # Actually no. The tensor T is in R^{n*m} x R^{m*p} x R^{p*n}
        # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise
        
        # Let's build the target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reconstruct from decomposition
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a tensor of order 3, can be computed via:
        # For each r, the outer product U[:,r] outer V[:,r] outer W[:,r]
        recon = np.zeros_like(target)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        return np.allclose(recon, target, atol=1e-10)
    
    # Try to find lower rank decompositions
    # Known: rank(3x3x4) = 20 (Strassen-type)
    # Actually, let me think more carefully.
    
    # For matrix multiplication m x n x p, the rank is the tensor rank of the matmul tensor.
    # For 3x3x3, Strassen gave rank 7.
    # For 3x3x4, we need to find the minimal rank.
    
    # Let's try a systematic approach:
    # The tensor has 36 non-zero entries (one for each (i,j,k) triple).
    # A rank-r decomposition means we can write T as sum of r rank-1 tensors.
    
    # Strategy: Use the fact that we can apply basis transformations.
    # Fix W in a canonical form (e.g., first r standard basis vectors if r <= p*n=12,
    # or a structured sparse form), then solve for U and V.
    
    # Actually, a better approach for small dimensions:
    # The matmul tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want min r such that T = sum_{s=1}^r u_s v_s w_s (outer products)
    
    # Key insight: We can restrict W to have a specific structure.
    # By the "slice rank" argument and known results, let's try rank 20.
    
    # Actually, let me try a different approach: use the known Strassen-like algorithm.
    # For 3x3x3, rank 7 is known. For 3x3x4, we can embed the 3x3x3 result.
    
    # Let me try to construct a rank-20 decomposition directly.
    # The idea: group the 36 terms into 20 groups using algebraic identities.
    
    # Alternative: use numerical optimization to find a low-rank decomposition.
    # Start with random initialization and use gradient descent on the non-convex problem.
    
    def try_rank(r):
        """Try to find a rank-r decomposition using numerical optimization."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)
        
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Use alternating least squares / gradient descent
        # The objective is ||T - sum_r u_r v_r w_r||^2
        # This is non-convex, so we use multiple restarts
        
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None
        
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each s, minimize ||target - sum_s u_s v_s w_s||^2
                # Fix V and W, solve for U
                # The residual is R = target - sum_{s != current} u_s v_s w_s
                # For u_s: minimize ||R + u_s v_s w_s||^2
                # This is a linear least squares problem for u_s
                # vec(R + u_s v_s w_s) = (W_s outer V_s) vec(u_s) + vec(R)
                # where W_s = W[:,s], V_s = V[:,s]
                
                # Actually, let's use a simpler gradient-based approach
                # Compute gradient w.r.t. U, V, W
                
                # T_recon[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
                # dT_recon/du[a,s] = V[b,s] W[c,s]
                # dT_recon/dv[b,s] = U[a,s] W[c,s]  
                # dT_recon/dw[c,s] = U[a,s] V[b,s]
                
                recon = np.zeros_like(target)
                for s in range(r):
                    recon += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
                
                residual = target - recon
                
                # Gradient for U: grad_U[a,s] = sum_{b,c} residual[a,b,c] * V[b,s] * W[c,s]
                # = einsum('abc,bs,cs->as', residual, V, W)
                grad_U = np.einsum('abc,bs,cs->as', residual, V, W)
                grad_V = np.einsum('abc,as,cs->bs', residual, U, W)
                grad_W = np.einsum('abc,as,bs->cs', residual, U, V)
                
                # Simple gradient descent with adaptive step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                loss = np.sum(residual**2)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-8:
                    break
        
        if best_loss < 1e-6:
            # Try to round to rational numbers
            U_r = np.round(best_U * 100) / 100
            V_r = np.round(best_V * 100) / 100
            W_r = np.round(best_W * 100) / 100
            
            recon = np.zeros_like(target)
            for s in range(r):
                recon += np.einsum('a,b,c->abc', U_r[:,s], V_r[:,s], W_r[:,s])
            
            if np.allclose(recon, target, atol=1e-6):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    # Try ranks from 20 upward
    for r in [20, 21, 22, 23, 24]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If numerical approach fails, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
