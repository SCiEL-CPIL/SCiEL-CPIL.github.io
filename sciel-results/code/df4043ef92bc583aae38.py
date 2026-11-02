"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Precompute the tensor T as a flat vector for faster operations
    # T_flat has length n*m*p*m*n = (n*m)*(m*p)*(p*n)
    # But we only need to work with the non-zero entries for efficiency
    # Actually, let's keep the 3D tensor but use efficient einsums
    
    # Helper function to check if a solution is valid
    def check_solution(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9)

    # Helper to round to half-integers and check
    def try_round_and_check(U, V, W, T):
        """Try rounding to half-integers and check validity."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if check_solution(U_r, V_r, W_r, T):
            return U_r, V_r, W_r, True
        return None, None, None, False

    # Gradient-based optimization for finding low-rank decompositions
    def gradient_optimize(r, T, U_init, V_init, W_init, max_iters=500, lr=0.01):
        """
        Use gradient descent to minimize ||T - sum_c u_c v_c w_c||^2
        with additional regularization to encourage sparsity/simple values.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute for efficiency
        # We'll use Adam-like optimization with momentum
        
        # Initialize momentum buffers
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        t = 0
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            t += 1
            
            # Compute current sum: S = sum_c u_c v_c w_c
            S = np.einsum('ic,jc,kc->ijk', U, V, W)
            residual = T - S
            
            # Compute loss
            loss = np.sum(residual ** 2)
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Check convergence
            if loss < 1e-12:
                break
            
            # Compute gradients
            # dL/dU = -2 * residual contracted with V, W
            # dU[i,c] = -2 * sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c]
            dU = -2 * np.einsum('ijk,jc,kc->ic', residual, V, W)
            dV = -2 * np.einsum('ijk,ic,kc->jc', residual, U, W)
            dW = -2 * np.einsum('ijk,ic,jc->kc', residual, U, V)
            
            # Add sparsity regularization (L1-like) to encourage simple values
            # This helps find solutions that are close to half-integers
            reg_lambda = 0.001
            dU += reg_lambda * np.sign(U)
            dV += reg_lambda * np.sign(V)
            dW += reg_lambda * np.sign(W)
            
            # Adam update
            mU = b1 * mU + (1 - b1) * dU
            mV = b1 * mV + (1 - b1) * dV
            mW = b1 * mW + (1 - b1) * dW
            
            vU = b2 * vU + (1 - b2) * dU ** 2
            vV = b2 * vV + (1 - b2) * dV ** 2
            vW = b2 * vW + (1 - b2) * dW ** 2
            
            mU_hat = mU / (1 - b1 ** t)
            mV_hat = mV / (1 - b1 ** t)
            mW_hat = mW / (1 - b1 ** t)
            
            vU_hat = vU / (1 - b2 ** t)
            vV_hat = vV / (1 - b2 ** t)
            vW_hat = vW / (1 - b2 ** t)
            
            # Learning rate schedule
            current_lr = lr / (1 + it * 0.01)
            
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Periodic restart: if loss increases, try a different initialization
            if it > 0 and it % 50 == 0:
                # Check if we're stuck
                S_check = np.einsum('ic,jc,kc->ijk', U, V, W)
                loss_check = np.sum((T - S_check) ** 2)
                if loss_check > best_loss * 1.1:
                    # Revert to best and try different perturbation
                    U = best_U + np.random.randn(n*m, r) * 0.01
                    V = best_V + np.random.randn(m*p, r) * 0.01
                    W = best_W + np.random.randn(p*n, r) * 0.01
        
        return best_U, best_V, best_W, best_loss

    # Power method for rank-1 approximation
    def power_method_rank1(T, u_init, v_init, w_init, num_iters=20):
        """Find best rank-1 approximation to T using power method."""
        u = u_init.copy()
        v = v_init.copy()
        w = w_init.copy()
        
        if np.linalg.norm(u) < 1e-6: u = np.random.randn(n*m)
        if np.linalg.norm(v) < 1e-6: v = np.random.randn(m*p)
        if np.linalg.norm(w) < 1e-6: w = np.random.randn(p*n)
        
        for _ in range(num_iters):
            u_new = np.einsum('ijk,j,k->i', T, v, w)
            v_new = np.einsum('ijk,i,k->j', T, u, w)
            w_new = np.einsum('ijk,i,j->k', T, u, v)
            
            norm_u = np.linalg.norm(u_new)
            norm_v = np.linalg.norm(v_new)
            norm_w = np.linalg.norm(w_new)
            
            if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                break
            
            u_new /= norm_u
            v_new /= norm_v
            w_new /= norm_w
            
            u, v, w = u_new, v_new, w_new
        
        # Scale to match
        uo = np.einsum('i,j,k->ijk', u, v, w)
        denom = np.sum(uo * uo)
        if denom < 1e-10:
            return u, v, w, 0.0
        s = np.sum(T * uo) / denom
        
        return s * u, s * v, s * w, s

    # Search for solutions with decreasing rank
    for r in range(59, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple strategies for this rank
        strategies = []
        
        # Strategy 1: Gradient-based optimization with random initialization
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
            
            U_init = np.random.randn(n*m, r) * 0.5
            V_init = np.random.randn(m*p, r) * 0.5
            W_init = np.random.randn(p*n, r) * 0.5
            
            U_opt, V_opt, W_opt, loss = gradient_optimize(
                r, T, U_init, V_init, W_init, max_iters=300, lr=0.02
            )
            
            # Try to round to half-integers
            U_r, V_r, W_r, valid = try_round_and_check(U_opt, V_opt, W_opt, T)
            if valid:
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                best_rank = r
                break
        
        if best_rank == r:
            continue
        
        # Strategy 2: Power method with structured initialization
        for restart in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with schoolbook-like structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1
            
            # Fill remaining with random values
            for c in range(col, r):
                U[:, c] = np.random.randn(n*m) * 0.3
                V[:, c] = np.random.randn(m*p) * 0.3
                W[:, c] = np.random.randn(p*n) * 0.3
            
            # Power method iterations
            current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
            converged = False
            
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                residual = T - current_sum
                residual_norm = np.linalg.norm(residual)
                
                if residual_norm < 1e-9:
                    converged = True
                    break
                
                # Select column to update
                c = np.random.randint(r)
                
                col_sum = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                target = residual + col_sum
                
                # Power method for this column
                u, v, w, s = power_method_rank1(target, U[:, c], V[:, c], W[:, c])
                
                U[:, c] = u
                V[:, c] = v
                W[:, c] = w
                
                new_col = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                current_sum = current_sum - col_sum + new_col
            
            if converged:
                U_r, V_r, W_r, valid = try_round_and_check(U, V, W, T)
                if valid:
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = r
                    break
        
        if best_rank == r:
            continue
        
        # Strategy 3: Hybrid - use gradient to find good structure, then power method to polish
        for restart in range(15):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Gradient phase
            U, V, W, loss = gradient_optimize(r, T, U, V, W, max_iters=200, lr=0.02)
            
            # Power method polish
            current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
            converged = False
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                residual = T - current_sum
                residual_norm = np.linalg.norm(residual)
                
                if residual_norm < 1e-9:
                    converged = True
                    break
                
                c = np.random.randint(r)
                col_sum = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                target = residual + col_sum
                
                u, v, w, s = power_method_rank1(target, U[:, c], V[:, c], W[:, c])
                
                U[:, c] = u
                V[:, c] = v
                W[:, c] = w
                
                new_col = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                current_sum = current_sum - col_sum + new_col
            
            if converged:
                U_r, V_r, W_r, valid = try_round_and_check(U, V, W, T)
                if valid:
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = r
                    break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
