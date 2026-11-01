"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Build target tensor T (sparse representation)
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def compute_error(U, V, W):
        """Compute squared error efficiently."""
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((P - T) ** 2)
        
    def verify_exact(U, V, W):
        """Verify exact decomposition using rational arithmetic."""
        # Convert to fractions for exact check
        Uf = [[Fraction(U[i, t]) for t in range(U.shape[1])] for i in range(U.shape[0])]
        Vf = [[Fraction(V[i, t]) for t in range(V.shape[1])] for i in range(V.shape[0])]
        Wf = [[Fraction(W[i, t]) for t in range(W.shape[1])] for i in range(W.shape[0])]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(Uf[a][t] * Vf[b][t] * Wf[c][t] for t in range(len(Uf[0])))
                    if val != 1:
                        return False
        # Check that all other entries are zero
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    # Check if (a,b,c) corresponds to a valid (i,j,k)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    is_valid = (i == i2 and j == j2 and k == k2)
                    if not is_valid:
                        val = sum(Uf[a][t] * Vf[b][t] * Wf[c][t] for t in range(len(Uf[0])))
                        if val != 0:
                            return False
        return True
        
    def verify_fast(U, V, W):
        """Fast verification using floating point with tight tolerance."""
        P = np.einsum('at,bt,ct->abc', U, V, W)
        if not np.allclose(P, T, atol=1e-10, rtol=1e-10):
            return False
        # Additional check: verify key entries exactly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(P[a, b, c] - 1.0) > 1e-10:
                        return False
        return True
        
    def schoolbook():
        r = n * m * p  # 24
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    best_result = schoolbook()
    best_rank = n * m * p  # 24
    
    # Candidates for entries - use a broader set including more values
    candidates = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=np.float64)
    
    # Try ranks from 17 upward (known rank for 2x3x4 is 17)
    for rank in range(17, best_rank):
        if time.time() > deadline - 20:
            break
            
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart * 7919 + rank * 131)
            
            # Strategy: start with sparse random initialization
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # Initialize with sparse non-zero entries
            for t in range(rank):
                # Randomly choose which entries to be non-zero
                n_nonzero_u = np.random.randint(1, 4)
                n_nonzero_v = np.random.randint(1, 4)
                n_nonzero_w = np.random.randint(1, 3)
                
                u_idx = np.random.choice(dim_u, size=n_nonzero_u, replace=False)
                v_idx = np.random.choice(dim_v, size=n_nonzero_v, replace=False)
                w_idx = np.random.choice(dim_w, size=n_nonzero_w, replace=False)
                
                for idx in u_idx:
                    U[idx, t] = np.random.choice(candidates)
                for idx in v_idx:
                    V[idx, t] = np.random.choice(candidates)
                for idx in w_idx:
                    W[idx, t] = np.random.choice(candidates)
            
            cur_err = compute_error(U, V, W)
            
            # Local search with coordinate descent
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if cur_err < 1e-12:
                    if verify_fast(U, V, W):
                        found = True
                        best_rank = rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
                    
                # Randomly pick a matrix, row, and column to modify
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                    old_val = U[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val:
                        continue
                    U[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        U[idx, t] = old_val
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                    old_val = V[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val:
                        continue
                    V[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        V[idx, t] = old_val
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    old_val = W[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val:
                        continue
                    W[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        W[idx, t] = old_val
                        
            if found:
                break
                
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
