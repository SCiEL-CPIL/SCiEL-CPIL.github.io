import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Target tensor T[i, j, k] = 1 if j == k else 0
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i, j, k]
    
    # We will search for rank R
    # Start with R=16, then 17, then 18
    
    best_solution = None
    best_rank = 18
    
    # Fallback: Standard rank 18
    U18 = np.zeros((n*m, 18))
    V18 = np.zeros((m*p, 18))
    W18 = np.zeros((p*n, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U18[i*m + j, col] = 1
                V18[j*p + k, col] = 1
                W18[k*n + i, col] = 1
                col += 1
    best_solution = {'U': U18.tolist(), 'V': V18.tolist(), 'W': W18.tolist()}
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s buffer
    
    # Helper to check if a real matrix can be snapped to half-integers and is exact
    def check_solution(U, V, W, R):
        # Snap to nearest 0.5
        Us = np.round(U * 2) / 2
        Vs = np.round(V * 2) / 2
        Ws = np.round(W * 2) / 2
        
        # Check exactness
        # Reconstruct tensor
        # T[i, j, k] = sum_r Us[i*m+j, r] Vs[j*p+k, r] Ws[k*n+i, r]
        
        # Efficient check:
        # For each i, j, k, compute sum.
        # Since dimensions are small, we can do it directly.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0 if j == k else 0.0
                    val = 0.0
                    for r in range(R):
                        val += Us[i*m + j, r] * Vs[j*p + k, r] * Ws[k*n + i, r]
                    if abs(val - target) > 1e-9:
                        return False, None, None, None
        return True, Us, Vs, Ws

    def try_rank(R):
        nonlocal best_solution, best_rank
        if R >= best_rank:
            return
        
        max_iter = 5000
        lr = 0.01
        decay = 0.999
        
        # Random restarts
        # We can do many restarts because each iter is fast for small R
        n_restarts = 50
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            
            # Initialize
            # Scale: entries are typically small.
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return
                
                # Compute gradients
                # L = 0.5 * sum_{i,j,k} (sum_r U_{im+j,r} V_{jp+k,r} W_{kn+i,r} - T_{ijk})^2
                # dL/dU_{im+j,r} = sum_{k} ( ... ) * V_{jp+k,r} * W_{kn+i,r}
                
                # We can vectorize this.
                # Let's construct the error tensor E[i, j, k]
                
                # E[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # Instead of full tensor, we can compute gradients per rank r
                # grad_U[i*m+j, r] = sum_k (E[i,j,k] - T[i,j,k]) * V[j*p+k, r] * W[k*n+i, r]
                # grad_V[j*p+k, r] = sum_i (E[i,j,k] - T[i,j,k]) * U[i*m+j, r] * W[k*n+i, r]
                # grad_W[k*n+i, r] = sum_j (E[i,j,k] - T[i,j,k]) * U[i*m+j, r] * V[j*p+k, r]
                
                # To compute E efficiently:
                # E[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                
                # Let's just loop over i, j, k to compute gradients, it's small.
                # n=2, m=3, p=3 -> 18 entries.
                # R=16.
                # Total ops ~ 18 * 16 * 3 (for grad) * 3 (for 3 vars) ~ 2600 ops per iter.
                # 5000 iters * 50 restarts * 2600 ops = 650M ops.
                # Python loop might be slow. 110s / 650M ~ 170ns/op. Might be tight.
                # Let's optimize with NumPy where possible.
                
                # Precompute indices?
                # Let's try a vectorized approach for E calculation.
                
                # U shape (nm, R)
                # V shape (mp, R)
                # W shape (pn, R)
                
                # We want E[i, j, k]
                # This is not a standard matrix product.
                
                # Alternative:
                # For each r, we have a rank-1 tensor U_r (nm,), V_r (mp,), W_r (pn,)
                # This doesn't map directly to i,j,k without reshaping.
                
                # Let's stick to the loop but make it tight.
                # Or, use the fact that T[i,j,k] is sparse (only j==k).
                
                # Let's calculate residuals R_ij_k = E_ij_k - T_ij_k
                # We can compute E by summing contributions.
                
                # Let's try to vectorize the gradient computation.
                # grad_U = zeros(nm, R)
                # grad_V = zeros(mp, R)
                # grad_W = zeros(pn, R)
                
                # For each i, j, k:
                # err = E[i,j,k] - T[i,j,k]
                # grad_U[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                # grad_V[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                # grad_W[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                
                # To get E[i,j,k], we need dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                # This is a sum over r.
                
                # We can precompute M_VW[i, j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                # But this is a 4D tensor.
                
                # Let's just do the loop. It's small enough.
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # We need E values.
                # We can compute E on the fly.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute E[i,j,k]
                            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # Vectorized dot product
                            e_val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            t_val = 1.0 if j == k else 0.0
                            err = e_val - t_val
                            
                            if abs(err) < 1e-10:
                                continue # Optimization: skip if error is tiny? No, need exact.
                                
                            # Update gradients
                            # grad_U[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                            v_w_prod = V[j*p+k, :] * W[k*n+i, :]
                            grad_U[i*m+j, :] += err * v_w_prod
                            
                            # grad_V[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                            u_w_prod = U[i*m+j, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += err * u_w_prod
                            
                            # grad_W[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                            u_v_prod = U[i*m+j, :] * V[j*p+k, :]
                            grad_W[k*n+i, :] += err * u_v_prod
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Periodically check for solution
                if it % 100 == 0:
                    is_sol, Us, Vs, Ws = check_solution(U, V, W, R)
                    if is_sol:
                        best_solution = {'U': Us.tolist(), 'V': Vs.tolist(), 'W': Ws.tolist()}
                        best_rank = R
                        return # Found solution for this rank, stop searching this rank? 
                              # Or keep looking for better? Rank is the score. Lower is better.
                              # If we found R=16, we are done.
                              # If we found R=17, we might still find R=16.
                              # So if R == 16, return. If R > 16, continue?
                              # Actually, if we find a solution for R, we can return it as best so far.
                              # But we want the MINIMAL rank.
                              # So if we find R=16, we are optimal.
                              # If we find R=17, we store it, but keep searching for 16.
                              # However, the search is per rank.
                              # So if try_rank(16) finds a solution, we return it.
                              # If try_rank(16) fails, we try try_rank(17).
                              # So inside try_rank, if we find a solution, we can return immediately.
                        pass
                
                # Decay learning rate
                lr *= decay
                
                # Early stop if loss is very small but not zero?
                # If loss < 1e-6, snap and check.
                if it % 10 == 0:
                    # Quick loss check
                    # Just check solution
                    is_sol, Us, Vs, Ws = check_solution(U, V, W, R)
                    if is_sol:
                        best_solution = {'U': Us.tolist(), 'V': Vs.tolist(), 'W': Ws.tolist()}
                        best_rank = R
                        return

    # Try ranks
    try_rank(16)
    if best_rank > 16:
        try_rank(17)
    
    return best_solution
