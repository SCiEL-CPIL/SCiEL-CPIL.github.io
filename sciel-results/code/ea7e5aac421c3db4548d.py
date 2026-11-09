"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
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
        T_rec = np.zeros((nm, mp, pn))
        for s in range(r):
            T_rec += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
        return np.max(np.abs(T_rec - T))
    
    for target_rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() > deadline - 20:
            break
        found = False
        n_attempts = 15
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            np.random.seed(attempt * 7919 + target_rank * 104729)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            for iteration in range(40):
                if time.time() > deadline - 20:
                    break
                M = np.einsum('bs,cs->sbc', V, W).reshape(target_rank, mp * pn)
                Y = T.reshape(nm, mp * pn)
                MMt = M @ M.T
                try:
                    U = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    U = Y @ M.T @ np.linalg.pinv(MMt)
                M = np.einsum('as,cs->sac', U, W).reshape(target_rank, nm * pn)
                Y = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                MMt = M @ M.T
                try:
                    V = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    V = Y @ M.T @ np.linalg.pinv(MMt)
                M = np.einsum('as,bs->sab', U, V).reshape(target_rank, nm * mp)
                Y = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                MMt = M @ M.T
                try:
                    W = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    W = Y @ M.T @ np.linalg.pinv(MMt)
            err = check_exact(U, V, W, target_rank)
            if err < 1e-9:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
            if err < 1e-3:
                for scale in [2, 4, 10]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    err_r = check_exact(U_r, V_r, W_r, target_rank)
                    if err_r < 1e-9:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
                if found:
                    break
        if found:
            break
    
    U = best_U[:, :best_rank]
    V = best_V[:, :best_rank]
    W = best_W[:, :best_rank]
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
