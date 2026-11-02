import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second buffer before 120s limit
    
    n, m, p = DIMENSIONS
    
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        return T

    T = build_tensor(n, m, p)
    
    best_solution = None
    best_rank = 11
    
    target_rank = 10
    max_iters = 500
    tol = 1e-10
    
    # Build rank-11 base solution for structured initialization
    def build_rank11():
        rank = 11
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_u(i, j, r, val): U[i*m+j, r] = val
        def set_v(j, k, r, val): V[j*p+k, r] = val
        def set_w(k, i, r, val): W[k*n+i, r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)
        
        return U, V, W

    U11, V11, W11 = build_rank11()
    
    def check_solution(U, V, W, rank, atol=1e-9):
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        return np.allclose(T_rec, T, atol=atol)
    
    def try_snap(U, V, W, rank):
        for denom in [2, 4, 6, 8, 12, 1, 3, 5]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if check_solution(U_s, V_s, W_s, rank):
                return (U_s, V_s, W_s)
        return None
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Different initialization strategies
        if r < 50:
            # Strategy 1: Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
        elif r < 100:
            # Strategy 2: Perturbed rank-11, drop one column and optimize
            # Try dropping each of the 11 columns and see if we can compensate
            drop_idx = r - 50
            U = U11[:, :drop_idx].copy()
            V = V11[:, :drop_idx].copy()
            W = W11[:, :drop_idx].copy()
            # Add one new random column to try to compensate
            U = np.hstack([U, np.random.randn(n*m, 1)])
            V = np.hstack([V, np.random.randn(m*p, 1)])
            W = np.hstack([W, np.random.randn(p*n, 1)])
        elif r < 150:
            # Strategy 3: Structured random with sparsity
            U = np.random.randn(n*m, target_rank) * np.random.choice([0, 1, -1], size=(n*m, target_rank))
            V = np.random.randn(m*p, target_rank) * np.random.choice([0, 1, -1], size=(m*p, target_rank))
            W = np.random.randn(p*n, target_rank) * np.random.choice([0, 1, -1], size=(p*n, target_rank))
            # Ensure not all zeros
            if np.allclose(U, 0): U += np.random.randn(n*m, target_rank)
            if np.allclose(V, 0): V += np.random.randn(m*p, target_rank)
            if np.allclose(W, 0): W += np.random.randn(p*n, target_rank)
        else:
            # Strategy 4: Very small perturbation of rank-11 with column merging
            # Take rank-11, randomly merge two columns into one
            U = U11.copy()
            V = V11.copy()
            W = W11.copy()
            i1 = np.random.randint(0, 11)
            i2 = np.random.randint(0, 11)
            if i1 != i2:
                # Merge column i2 into i1 with random weight
                w = np.random.uniform(0.5, 1.5)
                U[:, i1] += w * U[:, i2]
                V[:, i1] += w * V[:, i2]
                W[:, i1] += w * W[:, i2]
                # Remove column i2
                U = np.delete(U, i2, axis=1)
                V = np.delete(V, i2, axis=1)
                W = np.delete(W, i2, axis=1)
            # Add small random perturbation
            U += 0.1 * np.random.randn(n*m, target_rank)
            V += 0.1 * np.random.randn(m*p, target_rank)
            W += 0.1 * np.random.randn(p*n, target_rank)
        
        converged = False
        prev_err = np.inf
        stall_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            X_u = np.zeros((m*p*p*n, target_rank))
            for rr in range(target_rank):
                X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                U_new = U_new.T
            except:
                break
            U = U_new
            
            # Update V
            X_v = np.zeros((n*m*p*n, target_rank))
            for rr in range(target_rank):
                X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                V_new = V_new.T
            except:
                break
            V = V_new
            
            # Update W
            X_w = np.zeros((n*m*m*p, target_rank))
            for rr in range(target_rank):
                X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                W_new = W_new.T
            except:
                break
            W = W_new
            
            # Check convergence
            if it % 10 == 0 or it >= max_iters - 10:
                T_rec = np.zeros((n*m, m*p, p*n))
                for rr in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                err = np.linalg.norm(T - T_rec)
                
                if err < 1e-8:
                    converged = True
                    break
                if err < prev_err - 1e-12:
                    stall_count = 0
                else:
                    stall_count += 1
                prev_err = err
                
                if stall_count > 20:
                    break
        
        # Final check
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        err = np.linalg.norm(T - T_rec)
        
        if err < 1e-6:
            snapped = try_snap(U, V, W, target_rank)
            if snapped is not None:
                best_solution = snapped
                best_rank = target_rank
                break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return known rank 11 solution
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}
