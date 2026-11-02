import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def check_decomposition(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_rank(r):
        if time.time() > deadline:
            return None
            
        # Parameters: U (n*m*r), V (m*p*r), W (p*n*r)
        # Total params: r * (nm + mp + pn)
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        total_size = size_U + size_V + size_W
        
        # Random init
        x0 = np.random.randn(total_size) * 0.1
        
        def loss_and_grad(x):
            if time.time() > deadline:
                return 1e10, np.zeros(total_size)
                
            U = x[:size_U].reshape(n, m, r)
            V = x[size_U:size_U+size_V].reshape(m, p, r)
            W = x[size_U+size_V:].reshape(p, n, r)
            
            # Compute T_reconstructed
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Note: W is indexed by (k, i) in the product, but stored as (p, n, r) -> W[k, i, r]
            
            loss = 0.0
            # Gradient accumulation
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for rr in range(r):
                            val += U[i,j,rr] * V[j,k,rr] * W[k,i,rr]
                        err = val - 1.0
                        loss += err * err
                        
                        for rr in range(r):
                            # d(loss)/d(U[i,j,rr]) = 2 * err * V[j,k,rr] * W[k,i,rr]
                            grad_U[i,j,rr] += 2 * err * V[j,k,rr] * W[k,i,rr]
                            # d(loss)/d(V[j,k,rr]) = 2 * err * U[i,j,rr] * W[k,i,rr]
                            grad_V[j,k,rr] += 2 * err * U[i,j,rr] * W[k,i,rr]
                            # d(loss)/d(W[k,i,rr]) = 2 * err * U[i,j,rr] * V[j,k,rr]
                            grad_W[k,i,rr] += 2 * err * U[i,j,rr] * V[j,k,rr]
                            
            loss *= 0.5 # Cancel the 2
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                return None
                
            x0 = np.random.randn(total_size) * 0.1
            
            # Use L-BFGS-B
            # maxiter needs to be small to fit in time
            # We can't easily pass a time limit to minimize, so we rely on maxiter
            # and check time in callback? No, callback is for progress.
            # We will just run a fixed number of iterations (e.g., 100) and hope it converges or gets close.
            
            try:
                res = opt.minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'gtol': 1e-6})
            except Exception:
                continue
                
            if res.fun < 1e-5:
                x_opt = res.x
                U = x_opt[:size_U].reshape(n, m, r)
                V = x_opt[size_U:size_U+size_V].reshape(m, p, r)
                W = x_opt[size_U+size_V:].reshape(p, n, r)
                
                # Round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Convert to lists
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if check_decomposition(U_list, V_list, W_list, r):
                    return U_list, V_list, W_list
        return None

    # Search ranks
    # Start from 30 down to 20
    for r in range(30, 19, -1):
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res:
            U_list, V_list, W_list = res
            best_rank = r
            best_U, best_V, best_W = U_list, V_list, W_list
            break

    return {"U": best_U, "V": best_V, "W": best_W}
