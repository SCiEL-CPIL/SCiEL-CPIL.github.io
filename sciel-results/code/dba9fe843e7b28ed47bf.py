"""Matrix multiplication 3x3x5 - attempt to find a lower rank decomposition."""
import time
import random
import numpy as np
from itertools import product, combinations

N, M, P = 3, 3, 5
RANK_SCHOOLBOOK = N * M * P  # 45


def build_tensor():
    T = np.zeros((N * M, M * P, P * N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T


T = build_tensor()


def verify_decomposition(U, V, W, tol=1e-9):
    rank = U.shape[1]
    residual = np.zeros((N * M, M * P, P * N))
    for c in range(rank):
        residual += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(residual - T)) < tol


def schoolbook():
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _residual(U, V, W):
    return np.max(np.abs(np.einsum('ac,bc,dc->abd', U, V, W) - T))


def _try_one_rank(r, deadline, max_iters, rng):
    """Random + local search for a rank-r decomposition."""
    start = time.time()
    best_res = np.inf
    best_U = best_V = best_W = None
    # Candidate values for entries (integers and half-integers)
    cands = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    # Start from schoolbook and try to zero out random columns
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = rng.choice(cands, size=(N * M, r))
        V = rng.choice(cands, size=(M * P, r))
        W = rng.choice(cands, size=(P * N, r))
        
        res = _residual(U, V, W)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if res < 1e-9:
                break
        
        # Local refinement: try flipping one entry at a time
        if best_res < 1.0:
            improved = True
            while improved:
                improved = False
                # Randomly pick a matrix and position
                mat_idx = rng.randint(0, 2)
                if mat_idx == 0:
                    mat = best_U
                    size = (N * M, r)
                elif mat_idx == 1:
                    mat = best_V
                    size = (M * P, r)
                else:
                    mat = best_W
                    size = (P * N, r)
                
                row = rng.randint(0, size[0] - 1)
                col = rng.randint(0, size[1] - 1)
                old_val = mat[row, col]
                
                for new_val in cands:
                    if new_val == old_val:
                        continue
                    mat[row, col] = new_val
                    new_res = _residual(best_U, best_V, best_W)
                    if new_res < best_res:
                        best_res = new_res
                        improved = True
                        break
                    else:
                        mat[row, col] = old_val
                
                if best_res < 1e-9:
                    break
    
    if best_res < 1e-9:
        return best_U, best_V, best_W
    return None


def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 44 down to 40
    for r in range(44, 39, -1):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _try_one_rank(r, deadline - 2, max_iters=50000, rng=rng)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
