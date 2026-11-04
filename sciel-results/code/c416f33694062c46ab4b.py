import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety
    
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    best_result = None
    best_rank = n * m * p  # schoolbook rank 24

    def als_step(U, V, W, T, max_iters=100):
        rank = U.shape[1]
        for it in range(max_iters):
            # Update U
            M_U = np.einsum('abc,bl,cl->al', T, V, W)
            N_U = np.einsum('bl,cl,bm,cm->lm', V, W, V, W)
            try:
                U_new = (M_U.T @ np.linalg.inv(N_U)).T
            except np.linalg.LinAlgError:
                U_new = (M_U.T @ np.linalg.pinv(N_U)).T
            
            # Update V
            M_V = np.einsum('abc,al,cl->bl', T, U_new, W)
            N_V = np.einsum('al,cl,am,cm->lm', U_new, W, U_new, W)
            try:
                V_new = (M_V.T @ np.linalg.inv(N_V)).T
            except np.linalg.LinAlgError:
                V_new = (M_V.T @ np.linalg.pinv(N_V)).T
                
            # Update W
            M_W = np.einsum('abc,al,bl->cl', T, U_new, V_new)
            N_W = np.einsum('al,bl,am,bm->lm', U_new, V_new, U_new, V_new)
            try:
                W_new = (M_W.T @ np.linalg.inv(N_W)).T
            except np.linalg.LinAlgError:
                W_new = (M_W.T @ np.linalg.pinv(N_W)).T
                
            U, V, W = U_new, V_new, W_new
            
            # Early stopping check
            T_comp = np.einsum('al,bl,cl->abc', U, V, W)
            err = np.sum((T_comp - T)**2)
            if err < 1e-12:
                break
                
        return U, V, W

    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        found = False
        # ALS is fast, so we can try many attempts
        # Estimate 10-50ms per attempt?
        # 145s / 0.05s = ~2900 attempts possible.
        # Let's cap at 1000 per rank to be safe.
        max_attempts = 1000 
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale to avoid overflow/underflow in early iterations
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            U_opt, V_opt, W_opt = als_step(U, V, W, T, max_iters=50)
            
            # Check error
            T_comp = np.einsum('al,bl,cl->abc', U_opt, V_opt, W_opt)
            err = np.sum((T_comp - T)**2)
            
            if err < 1e-8:
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                T_snap = np.einsum('al,bl,cl->abc', U_snap, V_snap, W_snap)
                if np.allclose(T_snap, T, atol=1e-9):
                    best_result = (rank, U_snap, V_snap, W_snap)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        U_list = [[float(U[i][j]) for j in range(rank)] for i in range(nm)]
        V_list = [[float(V[i][j]) for j in range(rank)] for i in range(mp)]
        W_list = [[float(W[i][j]) for j in range(rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
