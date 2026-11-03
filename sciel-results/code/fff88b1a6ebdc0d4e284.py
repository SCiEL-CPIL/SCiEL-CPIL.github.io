import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8

    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0

    best_result = None
    best_rank = rank_schoolbook

    # Theoretical min rank is 19. We try 19 to 23.
    for r in range(19, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 30
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # ALS Iterations
            max_iters = 100
            converged = False
            prev_error = np.inf
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                # 1. Update U
                # M_U has shape (dim_W * dim_V, r)
                # M_U[a*dim_V+c, rr] = W[a, rr] * V[c, rr]
                M_U = np.zeros((dim_W * dim_V, r))
                for rr in range(r):
                    M_U[:, rr] = np.kron(W[:, rr], V[:, rr])
                    
                for b in range(dim_U):
                    y = T[:, b, :].flatten()
                    try:
                        u_b, _, _, _ = np.linalg.lstsq(M_U, y, rcond=None)
                        U[b, :] = u_b
                    except:
                        pass

                # 2. Update V
                # M_V has shape (dim_W * dim_U, r)
                # M_V[a*dim_U+b, rr] = W[a, rr] * U[b, rr]
                M_V = np.zeros((dim_W * dim_U, r))
                for rr in range(r):
                    M_V[:, rr] = np.kron(W[:, rr], U[:, rr])
                    
                for c in range(dim_V):
                    y = T[:, :, c].flatten()
                    try:
                        v_c, _, _, _ = np.linalg.lstsq(M_V, y, rcond=None)
                        V[c, :] = v_c
                    except:
                        pass

                # 3. Update W
                # M_W has shape (dim_U * dim_V, r)
                # M_W[b*dim_V+c, rr] = U[b, rr] * V[c, rr]
                M_W = np.zeros((dim_U * dim_V, r))
                for rr in range(r):
                    M_W[:, rr] = np.kron(U[:, rr], V[:, rr])
                    
                for a in range(dim_W):
                    y = T[a, :, :].flatten()
                    try:
                        w_a, _, _, _ = np.linalg.lstsq(M_W, y, rcond=None)
                        W[a, :] = w_a
                    except:
                        pass

                # Check convergence
                T_rec = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
                
                error = np.linalg.norm(T - T_rec)
                
                if error < 1e-8:
                    converged = True
                    break
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error

            if converged:
                # Try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rec_rounded = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec_rounded += np.einsum('a,b,c->abc', W_rounded[:, rr], U_rounded[:, rr], V_rounded[:, rr])
                
                error_rounded = np.linalg.norm(T - T_rec_rounded)
                
                if error_rounded < 1e-8:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    # If no low rank found, return schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
