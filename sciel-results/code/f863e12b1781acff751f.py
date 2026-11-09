import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140  # 140s deadline, leaving 10s margin
    
    def verify(U, V, W, r):
        """Verify that the decomposition exactly reconstructs the all-ones tensor."""
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += u[:, :, np.newaxis] * v[np.newaxis, :, :] * w.T[:, np.newaxis, :]
        return np.allclose(T, 1.0, atol=1e-6)
    
    def compute_T(U, V, W, r):
        """Compute the reconstructed tensor."""
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += u[:, :, np.newaxis] * v[np.newaxis, :, :] * w.T[:, np.newaxis, :]
        return T
    
    def objective_and_grad(params, r):
        """Objective and gradient for the tensor decomposition problem."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:].reshape(pn, r)
        
        T = compute_T(U, V, W, r)
        diff = 1.0 - T  # (n, m, p)
        
        error = np.sum(diff**2)
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)  # (n, m)
            v = V[:, r_idx].reshape(m, p)  # (m, p)
            w = W[:, r_idx].reshape(p, n)  # (p, n)
            
            # dU[i, j, r_idx] = -2 * sum_k diff[i, j, k] * v[j, k] * w[k, i]
            # A[i, j, k] = diff[i, j, k] * v[j, k]
            A = diff * v[np.newaxis, :, :]  # (n, m, p)
            # B[i, j] = sum_k A[i, j, k] * w[k, i]
            B = np.einsum('ijk,ki->ij', A, w)
            dU[:, r_idx] = (-2 * B).flatten()
            
            # dV[j, k, r_idx] = -2 * sum_i diff[i, j, k] * u[i, j] * w[k, i]
            C = diff * u[:, :, np.newaxis]  # (n, m, p)
            D = np.einsum('ijk,ki->jk', C, w)
            dV[:, r_idx] = (-2 * D).flatten()
            
            # dW[k, i, r_idx] = -2 * sum_j diff[i, j, k] * u[i, j] * v[j, k]
            E = diff * u[:, :, np.newaxis] * v[np.newaxis, :, :]  # (n, m, p)
            F = np.einsum('ijk->ik', E)  # (n, p)
            dW[:, r_idx] = (-2 * F.T).flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return error, grad
    
    # Schoolbook solution as fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_solution = None
    best_rank = n * m * p  # 60
    
    # Try ranks from 25 to 55
    for r in range(25, 56):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try multiple random restarts
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + restart * 7 + r)
            scale = 0.5
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B with gradient
                res = minimize(objective_and_grad, params, args=(r,), jac=True,
                              method='L-BFGS-B', 
                              options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round, V_round, W_round, r):
                        best_solution = (U_round, V_round, W_round)
                        best_rank = r
                        break
            except Exception:
                pass
        
        if best_solution and best_rank == r:
            break
    
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
