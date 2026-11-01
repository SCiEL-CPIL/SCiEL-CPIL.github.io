import numpy as np
import time
from fractions import Fraction

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
    deadline = 130  # Safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def gradient_descent(target_rank, max_iters=1000, lr=0.01, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = T - result
            
            # Gradients
            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
            
            # Update
            U -= lr * grad_U.reshape(n * m, target_rank)
            V -= lr * grad_V.reshape(m * p, target_rank)
            W -= lr * grad_W.reshape(p * n, target_rank)
            
            if iter_ % 100 == 0:
                err = compute_error(U, V, W, target_rank)
                if err < 1e-6:
                    break
                
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err

    def try_rank(target_rank):
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return False
            
            seed = restart * 1000 + target_rank
            U, V, W, error = gradient_descent(target_rank, max_iters=500, lr=0.05, seed=seed)
            
            if U is None:
                return False
                
            if error < 1e-5:
                # Polish
                for _ in range(100):
                    if time.time() - start_time > deadline:
                        return False
                    U_r = U.reshape(n, m, target_rank)
                    V_r = V.reshape(m, p, target_rank)
                    W_r = W.reshape(p, n, target_rank)
                    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    residual = T - result
                    err = np.linalg.norm(residual)
                    if err < 1e-8:
                        break
                    
                    grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
                    grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
                    grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
                    
                    U -= 0.01 * grad_U.reshape(n * m, target_rank)
                    V -= 0.01 * grad_V.reshape(m * p, target_rank)
                    W -= 0.01 * grad_W.reshape(p * n, target_rank)
                
                err = compute_error(U, V, W, target_rank)
                if err < 1e-6:
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-6:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
        return False

    # Try ranks from 14 to 17 (18 is schoolbook)
    for target_rank in range(14, R_school):
        if try_rank(target_rank):
            break

    # Final verification
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err

    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
