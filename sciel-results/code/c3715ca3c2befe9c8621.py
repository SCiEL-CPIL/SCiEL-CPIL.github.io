"""Rewritten program: search for a rank-minimal exact tensor decomposition of 2x3 * 3x9."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]
n, m, p = DIMENSIONS

def schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def tensor_residual(U, V, W):
    """Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] and return max deviation from 1."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
    return np.max(np.abs(T - 1.0))

def verify_exact(U, V, W, tol=1e-9):
    return tensor_residual(U, V, W) < tol

def greedy_eliminate(U, V, W, max_iter=100):
    """Try to eliminate columns by expressing one as a combination of others."""
    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    eliminated = 0
    for _ in range(max_iter):
        if rank <= 1:
            break
        best_col = -1
        best_err = float('inf')
        for c in range(rank):
            if rank == 1:
                break
            A = np.zeros((n*m*p, rank-1))
            b = np.zeros(n*m*p)
            idx = 0
            other_cols = [cc for cc in range(rank) if cc != c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        for oc_idx, oc in enumerate(other_cols):
                            A[idx, oc_idx] = U[i*m+j, oc] * V[j*p+k, oc] * W[k*n+i, oc]
                        idx += 1
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                A_alpha = A @ alpha
                err = np.max(np.abs(A_alpha - b))
                if err < 1e-6 and err < best_err:
                    best_err = err
                    best_col = c
            except:
                pass
        if best_col >= 0 and best_err < 1e-6:
            U = np.delete(U, best_col, axis=1)
            V = np.delete(V, best_col, axis=1)
            W = np.delete(W, best_col, axis=1)
            rank -= 1
            eliminated += 1
        else:
            break
    return U, V, W, eliminated

def construct():
    deadline = time.time() + 140
    
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Greedy elimination on schoolbook
    U_e, V_e, W_e, elim = greedy_eliminate(U, V, W, max_iter=200)
    if elim > 0 and verify_exact(U_e, V_e, W_e):
        best_rank = U_e.shape[1]
        best_U, best_V, best_W = U_e.copy(), V_e.copy(), W_e.copy()
    
    # Random restarts with local search
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        for init_rank in [best_rank, best_rank - 1, 25, 20, 15]:
            if time.time() > deadline:
                break
            if init_rank < 1:
                continue
            U_r = np.random.randint(-1, 2, size=(n*m, init_rank)).astype(float)
            V_r = np.random.randint(-1, 2, size=(m*p, init_rank)).astype(float)
            W_r = np.random.randint(-1, 2, size=(p*n, init_rank)).astype(float)
            
            # Local search: try to reduce rank
            U_ls, V_ls, W_ls, _ = greedy_eliminate(U_r, V_r, W_r, max_iter=50)
            if verify_exact(U_ls, V_ls, W_ls) and U_ls.shape[1] < best_rank:
                best_rank = U_ls.shape[1]
                best_U, best_V, best_W = U_ls.copy(), V_ls.copy(), W_ls.copy()
    
    U_list = [[float(v) for v in row] for row in best_U]
    V_list = [[float(v) for v in row] for row in best_V]
    W_list = [[float(v) for v in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
