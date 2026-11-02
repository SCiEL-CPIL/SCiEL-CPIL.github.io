import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def get_residuals(U, V, W):
        res = {}
        for (i, j, k) in triplets:
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[(i, j, k)] = val - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return np.sum([r*r for r in res.values()])

    def optimize(U, V, W, max_iters=100, tol=1e-8):
        # Simple Gradient Descent
        lr = 0.01
        for _ in range(max_iters):
            res = get_residuals(U, V, W)
            err = np.sum([r*r for r in res.values()])
            if err < tol:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k), r in res.items():
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                term_VW = V[vi] * W[wi]
                term_UW = U[ui] * W[wi]
                term_UV = U[ui] * V[vi]
                
                dU[ui] += 2 * r * term_VW
                dV[vi] += 2 * r * term_UW
                dW[wi] += 2 * r * term_UV
                
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return U, V, W

    def schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        return U, V, W

    def randomized_algebraic_search(max_restarts=50, max_iters=200):
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook()
        best_err = 0.0
        
        start_time = time.time()
        deadline = 100 # 50s margin
        
        # Try ranks from 40 to 48
        for r in range(40, 49):
            if time.time() > deadline:
                break
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize with schoolbook terms (first r)
                U_init, V_init, W_init = schoolbook(r)
                
                # Add small noise to help escape local minima
                noise_level = 0.1
                U_init += noise_level * np.random.randn(n*m, r)
                V_init += noise_level * np.random.randn(m*p, r)
                W_init += noise_level * np.random.randn(p*n, r)
                
                # Run optimization
                U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, max_iters=max_iters)
                
                err = total_error(U_opt, V_opt, W_opt)
                
                if err < 1e-6:
                    # Success!
                    return U_opt, V_opt, W_opt, r
                
                # Keep the best found so far (lowest error)
                if best_err == 0.0 or err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
        
        return best_U, best_V, best_W, best_rank

    U, V, W, rank = randomized_algebraic_search()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
