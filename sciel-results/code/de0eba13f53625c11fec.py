import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    nb = 3
    bs = 2
    rank_bs = nb * nb * nb * 7
    
    U = np.zeros((n*m, rank_bs), dtype=np.float64)
    V = np.zeros((m*p, rank_bs), dtype=np.float64)
    W = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    col += 1
                    
    # Verify initial decomposition
    def verify(U, V, W, tol=1e-9):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        U_reshaped = U.reshape(n, m, U.shape[1])
        V_reshaped = V.reshape(m, p, V.shape[1])
        W_reshaped = W.reshape(p, n, W.shape[1])
        U_A = np.einsum('ijr,ij->ir', U_reshaped, A)
        V_B = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, U_A, V_B)
        return np.allclose(C_computed, C_expected, atol=tol)
        
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_bs
    
    # Attempt rank reduction by merging pairs
    # This is a heuristic. We check if sum of two rank-1 terms is rank-1.
    # If so, replace them with the new rank-1 term.
    
    def is_rank_1(u, v, w, tol=1e-9):
        # u, v, w are vectors
        # Check if u v w is rank 1 tensor
        # This is always true for a single term.
        # We need to check if u1 v1 w1 + u2 v2 w2 is rank 1.
        pass

    # The pair merging is complex to implement correctly and quickly.
    # I will stick to the greedy removal of zero/redundant terms.
    
    # Greedy removal
    if time.time() < deadline - 20:
        removed = []
        for r in range(best_rank):
            if time.time() > deadline - 5:
                break
            mask = np.ones(best_rank, dtype=bool)
            mask[r] = False
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if verify(U_test, V_test, W_test):
                removed.append(r)
        
        if removed:
            mask = np.ones(best_rank, dtype=bool)
            mask[removed] = False
            best_U = U[:, mask]
            best_V = V[:, mask]
            best_W = W[:, mask]
            best_rank = best_rank - len(removed)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
