import numpy as np
import time
import itertools
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin

    def _verify_fast(U, V, W, tol=1e-9):
        """Faster verification using einsum."""
        r = U.shape[1]
        target = np.ones((n, m, p))
        U_3d = U.T.reshape(r, n, m)
        V_3d = V.T.reshape(r, m, p)
        W_3d = W.T.reshape(r, p, n)
        T_recon = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_recon, target, atol=tol)

    def _schoolbook(n, m, p):
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
        return U, V, W

    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63

    # --- Strategy: structured split with global mixing + joint optimization ---
    # Model: 3x3x7 = 3x3x3 core + 3x3x4 extension
    # Apply global linear transforms on j (input) and k (output) to mix blocks,
    # then jointly optimize the rank-r decomposition.
    
    def _structured_joint_search(r_target, max_iters=300, lr=0.02, n_restarts=3):
        """
        Joint optimization of rank-r decomposition using a structured initialization
        that mixes the 3x3x3 and 3x3x4 blocks via random linear transforms on j and k.
        """
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization:
            # Start from schoolbook-like structure but with random mixing
            # U: (n*m, r), V: (m*p, r), W: (p*n, r)
            
            # Random mixing transforms
            # Mix j-index (size m=3) with a random invertible 3x3 matrix
            J_mix = np.random.randn(m, m)
            np.linalg.cholesky(J_mix.T @ J_mix)  # ensure invertible (just for init variety)
            J_mix = J_mix + np.eye(m)  # make it clearly invertible
            
            # Mix k-index (size p=7) with a random invertible 7x7 matrix
            K_mix = np.random.randn(p, p)
            K_mix = K_mix + np.eye(p) * 10  # make well-conditioned
            
            # Initialize U, V, W with structured + random components
            # Core part (first 3 k-values) and extension part (last 4 k-values)
            
            U = np.zeros((n * m, r_target))
            V = np.zeros((m * p, r_target))
            W = np.zeros((p * n, r_target))
            
            # Fill with random values scaled appropriately
            U = np.random.randn(n * m, r_target) * 0.3
            V = np.random.randn(m * p, r_target) * 0.3
            W = np.random.randn(p * n, r_target) * 0.3
            
            # Add structured components
            for r_idx in range(min(r_target, n * m * p)):
                # Assign some structured pattern
                i = r_idx // (m * p)
                j = (r_idx // p) % m
                k = r_idx % p
                if i < n and j < m and k < p:
                    U[i * m + j, r_idx] += 1.0
                    V[j * p + k, r_idx] += 1.0
                    W[k * n + i, r_idx] += 1.0
            
            U_3d = U.T.reshape(r_target, n, m)
            V_3d = V.T.reshape(r_target, m, p)
            W_3d = W.T.reshape(r_target, p, n)
            target = np.ones((n, m, p))
            
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            def loss_and_grad(params):
                U_flat = params[0: r_target * n * m].reshape(r_target, n, m)
                V_flat = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_flat = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                T_recon = np.einsum('rij,rjk,rki->ijk', U_flat, V_flat, W_flat)
                diff = T_recon - target
                loss = np.sum(diff ** 2)
                
                dU = np.einsum('ijk,rjk,rki->rij', diff, V_flat, W_flat)
                dV = np.einsum('ijk,rij,rki->rjk', diff, U_flat, W_flat)
                dW = np.einsum('ijk,rij,rjk->rki', diff, U_flat, V_flat)
                
                grad = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                return loss, grad
            
            loss = np.inf
            cur_lr = lr
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                l, g = loss_and_grad(params)
                loss = l
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                if loss < 1e-6:
                    break
                # Adaptive learning rate
                params -= cur_lr * g
                cur_lr = max(0.001, cur_lr * 0.998)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
        
        if best_params is not None and best_loss < 1e-4:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            if _verify_fast(U_out, V_out, W_out, tol=1e-5):
                return U_out, V_out, W_out, best_loss
        
        return None

    # --- Pure random gradient search as fallback ---
    def _random_gradient_search(r_target, max_iters=200, lr=0.05, n_restarts=5):
        """Try to find a rank-r decomposition via random init + gradient descent."""
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, r_target) * 0.5
            V = np.random.randn(m * p, r_target) * 0.5
            W = np.random.randn(p * n, r_target) * 0.5
            
            U_3d = U.T.reshape(r_target, n, m)
            V_3d = V.T.reshape(r_target, m, p)
            W_3d = W.T.reshape(r_target, p, n)
            target = np.ones((n, m, p))
            
            params = np.concatenate([
                U_3d.reshape(-1),
                V_3d.reshape(-1),
                W_3d.reshape(-1)
            ])
            
            def loss_and_grad(params):
                U_flat = params[0: r_target * n * m].reshape(r_target, n, m)
                V_flat = params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
                W_flat = params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
                
                T_recon = np.einsum('rij,rjk,rki->ijk', U_flat, V_flat, W_flat)
                diff = T_recon - target
                loss = np.sum(diff ** 2)
                
                dU = np.einsum('ijk,rjk,rki->rij', diff, V_flat, W_flat)
                dV = np.einsum('ijk,rij,rki->rjk', diff, U_flat, W_flat)
                dW = np.einsum('ijk,rij,rjk->rki', diff, U_flat, V_flat)
                
                grad = np.concatenate([
                    dU.reshape(-1),
                    dV.reshape(-1),
                    dW.reshape(-1)
                ])
                return loss, grad
            
            loss = np.inf
            cur_lr = lr
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                l, g = loss_and_grad(params)
                loss = l
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                if loss < 1e-6:
                    break
                params -= cur_lr * g
                cur_lr = max(0.001, cur_lr * 0.998)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
        
        if best_params is not None and best_loss < 1e-4:
            U_3d = best_params[0: r_target * n * m].reshape(r_target, n, m)
            V_3d = best_params[r_target * n * m : 2 * r_target * n * m].reshape(r_target, m, p)
            W_3d = best_params[2 * r_target * n * m : 3 * r_target * n * m].reshape(r_target, p, n)
            
            U_out = U_3d.transpose(1, 2, 0).reshape(n * m, r_target)
            V_out = V_3d.transpose(1, 2, 0).reshape(m * p, r_target)
            W_out = W_3d.transpose(1, 2, 0).reshape(p * n, r_target)
            
            if _verify_fast(U_out, V_out, W_out, tol=1e-5):
                return U_out, V_out, W_out, best_loss
        
        return None

    # --- Search over candidate ranks ---
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try ranks from 62 down to 22
    
    candidate_ranks = list(range(62, 21, -1))
    
    for r_target in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try structured joint search first
        if time.time() < deadline - 25:
            result = _structured_joint_search(r_target, max_iters=150, lr=0.03, n_restarts=2)
            if result is not None:
                U_best, V_best, W_best = result[0], result[1], result[2]
                best_rank = r_target
                found = True
        
        # Try random gradient search
        if not found and time.time() < deadline - 15:
            result = _random_gradient_search(r_target, max_iters=150, lr=0.05, n_restarts=3)
            if result is not None:
                U_best, V_best, W_best = result[0], result[1], result[2]
                best_rank = r_target
                found = True
        
        if found:
            break

    # Final check and return
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
