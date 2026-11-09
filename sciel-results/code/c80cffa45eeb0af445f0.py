# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the matrix multiplication tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p
    best_U = np.zeros((nm, schoolbook_rank))
    best_V = np.zeros((mp, schoolbook_rank))
    best_W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    best_rank = schoolbook_rank
    deadline = time.time() + 130

    def check_exact(U, V, W, r):
        T_rec = np.einsum('is,js,ks->ijk', U, V, W)
        return np.max(np.abs(T_rec - T))

    def round_and_check(U, V, W, r):
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r, r) < 1e-9:
                return U_r, V_r, W_r
        return None

    def als_step(U, V, W, T, r, max_iters=200, tol=1e-12):
        for _ in range(max_iters):
            # Update U
            # M[s, j*p+k] = V[j, s] * W[k, s]
            M = np.einsum('js,ks->sjsk', V, W).reshape(r, mp * pn)
            Y = T.reshape(nm, mp * pn)
            MMt = M @ M.T
            YMt = Y @ M.T
            try:
                U_new = np.linalg.solve(MMt, YMt).T
            except:
                U_new = np.linalg.lstsq(MMt, YMt, rcond=None)[0].T
            
            # Update V
            M = np.einsum('is,ks->sisk', U_new, W).reshape(r, nm * pn)
            Y = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            MMt = M @ M.T
            YMt = Y @ M.T
            try:
                V_new = np.linalg.solve(MMt, YMt).T
            except:
                V_new = np.linalg.lstsq(MMt, YMt, rcond=None)[0].T
            
            # Update W
            M = np.einsum('is,js->sij', U_new, V_new).reshape(r, nm * mp)
            Y = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            MMt = M @ M.T
            YMt = Y @ M.T
            try:
                W_new = np.linalg.solve(MMt, YMt).T
            except:
                W_new = np.linalg.lstsq(MMt, YMt, rcond=None)[0].T
            
            # Check convergence
            if np.max(np.abs(U_new - U)) < tol and np.max(np.abs(V_new - V)) < tol and np.max(np.abs(W_new - W)) < tol:
                break
            U, V, W = U_new, V_new, W_new
        
        return U, V, W

    def try_rank(target_rank, current_U, current_V, current_W, current_rank, n_attempts=30):
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None
            
            if attempt == 0 and current_rank == target_rank + 1:
                # Warm start: Drop one random column from current solution
                drop_idx = np.random.randint(current_rank)
                keep = [i for i in range(current_rank) if i != drop_idx]
                U = current_U[:, keep].copy()
                V = current_V[:, keep].copy()
                W = current_W[:, keep].copy()
            else:
                # Random initialization
                np.random.seed(attempt * 1000 + target_rank)
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
            
            # Run ALS
            U, V, W = als_step(U, V, W, T, target_rank)
            
            # Check and round
            result = round_and_check(U, V, W, target_rank)
            if result:
                return result
        
        return None

    # Try to find lower rank decompositions
    cur_U, cur_V, cur_W, cur_rank = best_U, best_V, best_W, best_rank
    
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank // 2), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, cur_U, cur_V, cur_W, cur_rank, n_attempts=30)
        if result is not None:
            cur_U, cur_V, cur_W = result
            cur_rank = target_rank
        # If result is None, we keep the previous best (cur_U, cur_V, cur_W, cur_rank)
        # and try the next lower rank.

    U = cur_U[:, :cur_rank]
    V = cur_V[:, :cur_rank]
    W = cur_W[:, :cur_rank]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
