import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec - 1.0

    def error_and_grad(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        err = compute_error(U, V, W, rank)
        loss = np.sum(err**2)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            grad_U[:, r] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w, optimize=True).flatten()
            grad_V[:, r] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w, optimize=True).flatten()
            grad_W[:, r] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v, optimize=True).flatten()
            
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad_flat

    def try_snap_and_validate(U, V, W, rank, atol=1e-8):
        U1 = np.round(U * 2) / 2
        V1 = np.round(V * 2) / 2
        W1 = np.round(W * 2) / 2
        if is_valid(U1, V1, W1, rank, atol):
            return U1, V1, W1
        
        U2 = np.round(U)
        V2 = np.round(V)
        W2 = np.round(W)
        if is_valid(U2, V2, W2, rank, atol):
            return U2, V2, W2
            
        return None, None, None

    def search_decomposition(target_rank, time_budget, num_restarts=10):
        if time.time() > deadline:
            return None, None, None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None, None, None
                
            rng = np.random.RandomState(restart)
            U_init = rng.randn(n * m, target_rank) * 0.1
            V_init = rng.randn(m * p, target_rank) * 0.1
            W_init = rng.randn(p * n, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(error_and_grad, params_init, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            except Exception:
                continue
                
            if res.fun < 1e-10:
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                U_opt = res.x[:size_U].reshape(n * m, target_rank)
                V_opt = res.x[size_U:size_U + size_V].reshape(m * p, target_rank)
                W_opt = res.x[size_U + size_V:].reshape(p * n, target_rank)
                
                U_s, V_s, W_s = try_snap_and_validate(U_opt, V_opt, W_opt, target_rank)
                if U_s is not None:
                    return U_s, V_s, W_s
                    
        return None, None, None

    for rank in range(25, 45):
        if time.time() > deadline:
            break
            
        U, V, W = search_decomposition(rank, 10.0, num_restarts=5)
        
        if U is not None:
            best_solution = {"U": U, "V": V, "W": W}
            best_rank = rank
            break

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
