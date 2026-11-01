import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    def compute_tensor(U, V, W, rank):
        """Compute the tensor from decomposition."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    def compute_error(U, V, W, rank):
        """Compute the error of the tensor decomposition."""
        result = compute_tensor(U, V, W, rank)
        return np.linalg.norm(result - T)
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition with tight tolerance."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_rank(target_rank, max_restarts=30):
        """Try to find a decomposition of given rank."""
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return None, None, None, None
            
            rng = np.random.RandomState(restart * 7919 + target_rank * 104729 + 12345)
            
            # Structured initialization: start with sparse patterns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random sparse patterns from {0, 1, -1}
            for r in range(target_rank):
                # Each column gets a sparse pattern
                u_mask = rng.rand(n * m) < 0.4
                v_mask = rng.rand(m * p) < 0.4
                w_mask = rng.rand(p * n) < 0.4
                
                U[u_mask, r] = rng.choice([1, -1], size=np.sum(u_mask))
                V[v_mask, r] = rng.choice([1, -1], size=np.sum(v_mask))
                W[w_mask, r] = rng.choice([1, -1], size=np.sum(w_mask))
            
            # Add small random perturbation to break symmetry
            U += rng.randn(n * m, target_rank) * 0.05
            V += rng.randn(m * p, target_rank) * 0.05
            W += rng.randn(p * n, target_rank) * 0.05
            
            # ALS iterations with early stopping
            prev_error = np.inf
            for iter_ in range(150):
                if time.time() - start_time > deadline:
                    return None, None, None, None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((target_rank, p))
                        for k in range(p):
                            A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                        except:
                            pass
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                        except:
                            pass
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for j in range(m):
                            A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                        except:
                            pass
                
                error = compute_error(U, V, W, target_rank)
                
                if error < 1e-10:
                    break
                
                if error > prev_error * 1.001 and iter_ > 30:
                    break
                prev_error = error
            
            if prev_error < 1e-8:
                # Polish with gradient descent
                for polish_iter in range(100):
                    if time.time() - start_time > deadline:
                        break
                    
                    result = compute_tensor(U, V, W, target_rank)
                    residual = T - result
                    
                    if np.linalg.norm(residual) < 1e-12:
                        break
                    
                    # Gradient step
                    lr = 1e-4
                    for r in range(target_rank):
                        for i in range(n):
                            for j in range(m):
                                grad_U = sum(residual[i, j, k] * V[j * p + k, r] * W[k * n + i, r] for k in range(p))
                                U[i * m + j, r] -= lr * grad_U
                                
                        for j in range(m):
                            for k in range(p):
                                grad_V = sum(residual[i, j, k] * U[i * m + j, r] * W[k * n + i, r] for i in range(n))
                                V[j * p + k, r] -= lr * grad_V
                                
                        for k in range(p):
                            for i in range(n):
                                grad_W = sum(residual[i, j, k] * U[i * m + j, r] * V[j * p + k, r] for j in range(m))
                                W[k * n + i, r] -= lr * grad_W
                    
                    if compute_error(U, V, W, target_rank) < 1e-12:
                        break
                
                # Final verification
                max_err = verify_exact(U, V, W, target_rank)
                if max_err < 1e-6:
                    return U, V, W, max_err
        
        return None, None, None, None
    
    # Try ranks from 14 upward
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
        
        max_restarts = 40 if target_rank == 14 else 20
        U, V, W, err = try_rank(target_rank, max_restarts)
        
        if U is not None and err < 1e-6:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # Final verification
    if best_rank < R_school:
        final_err = verify_exact(best_U, best_V, best_W, best_rank)
        if final_err > 1e-4:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
