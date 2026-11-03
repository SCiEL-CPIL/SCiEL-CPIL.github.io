"""
Matrix multiplication tensor decomposition for M_{2,3,10}.
Uses structured initialization + ALS with term merging heuristics.
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _als_step(U, V, W, T, n, m, p, max_iters=200):
    """Run ALS iterations to refine U, V, W."""
    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.reshape(m * p, -1)
    T_flat_w = T.reshape(p * n, -1)
    
    for it in range(max_iters):
        # Update U
        M = np.einsum('br,cr->bcr', V, W)
        M_flat = M.transpose(2, 0, 1).reshape(-1, rank) if False else None
        # Proper reshaping: M is (m*p, p*n, rank)
        # We want for each r, a vector of length m*p*p*n
        # M_flat[r, b*(p*n) + c] = M[b, c, r]
        M_flat = M.transpose(2, 0, 1).reshape(M.shape[2], -1)  # (rank, m*p*p*n)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat_u.T, rcond=None)[0].T
        except:
            pass
        
        # Update V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(N.shape[2], -1)  # (rank, n*m*p*n)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            pass
        
        # Update W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(P.shape[2], -1)  # (rank, n*m*m*p)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-10:
            break
    
    return U, V, W

def _try_rank(n, m, p, rank, seed=42, max_iters=500):
    """Try to find a rank-r decomposition."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # Structured initialization: use the natural basis
    # Start from trivial and try to compress
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    if rank >= n * m * p:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, True
    
    # Random initialization with small values
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    U, V, W = _als_step(U, V, W, T, n, m, p, max_iters=max_iters)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _get_trivial_decomposition(n, m, p):
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = 30
    # Try ranks from 30 upward
    
    best_result = _get_trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try a few ranks with multiple seeds
    ranks_to_try = [30, 31, 32, 35, 40, 50]
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        num_seeds = 5 if rank <= 35 else 3
        max_iters = 300 if rank <= 35 else 150
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U, V, W, success = _try_rank(n, m, p, rank, seed=seed, max_iters=max_iters)
            if success:
                # Convert to lists
                U_list = [[float(val) for val in row] for row in U]
                V_list = [[float(val) for val in row] for row in V]
                W_list = [[float(val) for val in row] for row in W]
                
                # Verify
                T = _build_tensor(n, m, p)
                U_chk = np.array(U_list)
                V_chk = np.array(V_list)
                W_chk = np.array(W_list)
                T_recon = np.einsum('ar,br,cr->abc', U_chk, V_chk, W_chk)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            # Found a solution at this rank, try lower
            continue
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
