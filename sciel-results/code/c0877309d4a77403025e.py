import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Build the 2x3x3 tensor
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for l in range(3):
                T_block[i*3+j, j*3+l, l*2+i] = 1.0

    def try_block_decomposition(r_block, max_trials=80):
        """Try to find rank-r_block decomposition of 2x3x3 tensor"""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            seed = 50000 + r_block * 1000 + trial
            np.random.seed(seed)
            
            V = np.random.randn(9, r_block)
            W = np.random.randn(6, r_block)
            
            M = np.zeros((54, r_block))
            for r_idx in range(r_block):
                M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
            
            if np.linalg.matrix_rank(M) < r_block:
                continue
            
            U = np.zeros((6, r_block))
            valid = True
            for a in range(6):
                target = T_block[a, :, :].flatten()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_approx, T_block, atol=1e-9):
                return U, V, W
        
        return None

    # Try Kronecker-based: find rank-r for 2x3x3, get rank 3r for full
    found = False
    for r_block in [9, 10, 11, 12, 13]:
        if time.time() > deadline:
            break
        result = try_block_decomposition(r_block, max_trials=100)
        if result is not None:
            U_block, V_block, W_block = result
            rank_full = 3 * r_block
            if rank_full < best_rank:
                U_full = np.zeros((6, rank_full))
                V_full = np.zeros((27, rank_full))
                W_full = np.zeros((18, rank_full))
                
                for l in range(3):
                    for r_idx in range(r_block):
                        R = l * r_block + r_idx
                        U_full[:, R] = U_block[:, r_idx]
                        for j in range(3):
                            for s in range(3):
                                k = 3*l + s
                                V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        for s in range(3):
                            for i in range(2):
                                k = 3*l + s
                                W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                
                if verify_solution(U_full, V_full, W_full, T):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = rank_full
                    found = True
                    break
            break

    # Direct search on full tensor if Kronecker didn't help
    if not found:
        for r in [27, 28, 29, 30, 31]:
            if time.time() > deadline:
                break
            for trial in range(25):
                if time.time() > deadline:
                    break
                np.random.seed(90000 + r * 1000 + trial)
                
                V = np.random.randn(27, r)
                W = np.random.randn(18, r)
                
                M = np.zeros((486, r))
                for r_idx in range(r):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx]).flatten()
                
                if np.linalg.matrix_rank(M) < r:
                    continue
                
                U = np.zeros((6, r))
                valid = True
                for a in range(6):
                    target = T[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if verify_solution(U, V, W, T):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
            if best_rank == r:
                break

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
