"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix.
    
    Strategy: Try multiple approaches:
    1. Direct optimization on the full 2x5x5 tensor
    2. Block-diagonal approach using 5x5 decompositions
    3. Shared-V approach where both blocks share some V columns
    
    Fallback: schoolbook decomposition with rank 50.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook fallback
    def make_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
        return np.max(np.abs(Rec - T)) < tol
    
    def polish_solution(U, V, W, max_iter=2000):
        """High-precision polish pass using L-BFGS-B."""
        r = U.shape[1]
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_ = x[:n*m*r].reshape(n*m, r)
            V_ = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_ = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
            Res = T - Rec
            return np.sum(Res * Res)
        
        def gradient(x):
            U_ = x[:n*m*r].reshape(n*m, r)
            V_ = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_ = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
            Res = T - Rec
            Grad_U = np.einsum('ijl,jc,lc->ic', Res, V_, W_)
            Grad_V = np.einsum('ijl,ic,lc->jc', Res, U_, W_)
            Grad_W = np.einsum('ijl,ic,jc->lc', Res, U_, V_)
            return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15})
            if res.fun < 1e-12:
                U_new = res.x[:n*m*r].reshape(n*m, r)
                V_new = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_new = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                if check_solution(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        except Exception:
            pass
        return U, V, W
    
    def try_direct_decomposition(target_rank, max_trials=5):
        """Try direct optimization on the full 2x5x5 tensor."""
        if time.time() > deadline - 10:
            return None
        
        r = target_rank
        for trial in range(max_trials):
            if time.time() > deadline - 8:
                break
            
            np.random.seed(123 + r * 1000 + trial)
            U0 = np.random.randn(n*m, r) * 0.5
            V0 = np.random.randn(m*p, r) * 0.5
            W0 = np.random.randn(p*n, r) * 0.5
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U_ = x[:n*m*r].reshape(n*m, r)
                V_ = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_ = x[n*m*r + m*p*r :].reshape(p*n, r)
                Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
                Res = T - Rec
                return np.sum(Res * Res)
            
            def gradient(x):
                U_ = x[:n*m*r].reshape(n*m, r)
                V_ = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_ = x[n*m*r + m*p*r :].reshape(p*n, r)
                Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
                Res = T - Rec
                Grad_U = np.einsum('ijl,jc,lc->ic', Res, V_, W_)
                Grad_V = np.einsum('ijl,ic,lc->jc', Res, U_, W_)
                Grad_W = np.einsum('ijl,ic,jc->lc', Res, U_, V_)
                return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 800, 'gtol': 1e-10, 'ftol': 1e-14})
                
                if res.fun < 1e-10:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    if check_solution(U, V, W):
                        U, V, W = polish_solution(U, V, W)
                        if check_solution(U, V, W):
                            return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        
        return None
    
    def try_block_decomposition(target_rank_5x5, max_trials=10):
        """Try block-diagonal approach using 5x5 decompositions."""
        if time.time() > deadline - 15:
            return None
        
        r = target_rank_5x5
        T_5x5 = np.zeros((5, 25, 5))
        for j in range(5):
            for k in range(5):
                T_5x5[j, j*5 + k, k] = 1.0
        
        for trial in range(max_trials):
            if time.time() > deadline - 12:
                break
            
            np.random.seed(456 + r * 100 + trial)
            U0 = np.random.randn(5, r) * 0.4
            V0 = np.random.randn(25, r) * 0.4
            W0 = np.random.randn(5, r) * 0.4
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def obj_5x5(x):
                U_ = x[:5*r].reshape(5, r)
                V_ = x[5*r : 5*r + 25*r].reshape(25, r)
                W_ = x[5*r + 25*r :].reshape(5, r)
                Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
                Res = T_5x5 - Rec
                return np.sum(Res * Res)
            
            def grad_5x5(x):
                U_ = x[:5*r].reshape(5, r)
                V_ = x[5*r : 5*r + 25*r].reshape(25, r)
                W_ = x[5*r + 25*r :].reshape(5, r)
                Rec = np.einsum('ik,jk,lk->ijl', U_, V_, W_)
                Res = T_5x5 - Rec
                Grad_U = np.einsum('ijl,jc,lc->ic', Res, V_, W_)
                Grad_V = np.einsum('ijl,ic,lc->jc', Res, U_, W_)
                Grad_W = np.einsum('ijl,ic,jc->lc', Res, U_, V_)
                return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
            
            try:
                res = minimize(obj_5x5, x0, jac=grad_5x5, method='L-BFGS-B',
                              options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14})
                
                if res.fun < 1e-10:
                    U_5 = res.x[:5*r].reshape(5, r)
                    V_5 = res.x[5*r : 5*r + 25*r].reshape(25, r)
                    W_5 = res.x[5*r + 25*r :].reshape(5, r)
                    
                    # Verify 5x5 decomposition
                    Rec_5 = np.einsum('ik,jk,lk->ijl', U_5, V_5, W_5)
                    if np.max(np.abs(Rec_5 - T_5x5)) < 1e-8:
                        # Build full 2x5x5 decomposition
                        U = np.zeros((10, 2 * r))
                        V = np.zeros((25, 2 * r))
                        W = np.zeros((10, 2 * r))
                        
                        U[0:5, 0:r] = U_5
                        V[0:25, 0:r] = V_5
                        W[0:5, 0:r] = W_5
                        
                        U[5:10, r:2*r] = U_5
                        V[0:25, r:2*r] = V_5
                        W[5:10, r:2*r] = W_5
                        
                        if check_solution(U, V, W):
                            U, V, W = polish_solution(U, V, W)
                            if check_solution(U, V, W):
                                return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        
        return None
    
    # Try direct decomposition first (may find better solutions)
    for target_rank in [30, 32, 35, 38, 40]:
        if time.time() > deadline - 20:
            break
        
        result = try_direct_decomposition(target_rank)
        if result is not None:
            rank = result["U"].shape[1]
            if rank < best_rank:
                best_result = {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
                best_rank = rank
                print(f"Direct decomposition found with rank {rank}")
                break
    
    # Try block decomposition
    for target_rank_5x5 in [15, 16, 17, 18, 20, 22]:
        if time.time() > deadline - 10:
            break
        
        result = try_block_decomposition(target_rank_5x5)
        if result is not None:
            rank = result["U"].shape[1]
            if rank < best_rank:
                best_result = {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
                best_rank = rank
                print(f"Block decomposition found with rank {rank}")
                break
    
    return best_result
# EVOLVE-BLOCK-END
