import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target_rank = 50  # Schoolbook fallback
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)

    # Precompute index mappings for verification
    # u -> (i, j), v -> (j, k), w -> (k, i)
    # We need to check if res[i,j,k] == 1 if i,j,k match, else 0.
    # Actually, we can just build the target tensor once.
    target_tensor = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1
    
    # Precompute indices for einsum
    # U: (nm, r) -> (n, m, r)
    # V: (mp, r) -> (m, p, r)
    # W: (pn, r) -> (p, n, r)
    # Result: (n, m, p) -> reshape to (nm, mp, pn)?
    # Careful with reshape order.
    # U_reshaped[i, j, t] = U[i*m+j, t]
    # V_reshaped[j, k, t] = V[j*p+k, t]
    # W_reshaped[k, i, t] = W[k*n+i, t]
    # S[i, j, k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # We need S to match target_tensor reshaped?
    # target_tensor is indexed by (u, v, w).
    # u = i*m+j, v = j*p+k, w = k*n+i.
    # So target_tensor[u, v, w] corresponds to S[i, j, k].
    
    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # S[i, j, k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        S = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        
        # Reshape S to (nm, mp, pn) to compare with target_tensor
        # S is (n, m, p).
        # We need to map (i, j, k) to (u, v, w).
        # This is not a simple reshape because the indices are mixed.
        # However, we can just check the values directly.
        # Flatten S? No, the order is different.
        # Let's construct the expected S in the (n,m,p) domain.
        target_3d = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_3d[i, j, k] = 1
        
        # Check if S matches target_3d
        if not np.allclose(S, target_3d, atol=1e-9):
            return False
        
        # Check for "zero elsewhere"?
        # The einsum computes S[i,j,k] for all i,j,k.
        # The condition "zero elsewhere" in the flattened tensor corresponds to
        # S[i,j,k] being 0 if the indices don't match?
        # No, S[i,j,k] is ALWAYS computed for valid i,j,k.
        # The "zero elsewhere" condition in the flattened tensor (u,v,w)
        # corresponds to cases where u,v,w do NOT form a consistent (i,j,k).
        # But S is defined ONLY for consistent (i,j,k).
        # Wait, does the decomposition produce non-zero values for inconsistent indices?
        # The tensor T is defined on (u,v,w).
        # T[u,v,w] = sum_t U[u,t] V[v,t] W[w,t].
        # If u,v,w are inconsistent (e.g., j_u != j_v), this sum is still computed.
        # We require this sum to be 0.
        # My einsum S[i,j,k] only covers consistent indices.
        # I need to check the inconsistent ones too.
        
        # To do this efficiently, I can't easily use the 3D structure for inconsistent indices.
        # However, I can check the full flattened tensor.
        # Res = np.einsum('ut,vt,wt->uv', U, V, W) ? No, that's for 2D.
        # Res[u,v,w] = sum_t U[u,t] V[v,t] W[w,t].
        # This is a 3D tensor.
        # Res = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        Res = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        if not np.allclose(Res, target_tensor, atol=1e-9):
            return False
            
        return True

    def solve_for_W(U, V):
        r = U.shape[1]
        W_sol = np.zeros((pn, r), dtype=float)
        
        # Precompute UV products?
        # For each w, A_w is formed by U[u,t] * V[v,t] for all u,v.
        # We can compute UV_t = U[:,t] * V[:,t] (outer product) for each t?
        # No, A_w[idx, t] = U[u, t] * V[v, t].
        # This is the element-wise product of U and V broadcasted.
        # Let's create a tensor UV_t[u, v] = U[u, t] * V[v, t].
        # Shape: (r, nm, mp).
        UV_t = np.einsum('ut,vt->tuv', U, V)
        
        # b_w is the target vector for that w.
        # b_w[u, v] = 1 if u,v,w form consistent indices, else 0.
        
        for w in range(pn):
            k, i = divmod(w, n)
            # Target b for this w
            # b[u, v] = 1 if u=(i,j) and v=(j,k)
            # i.e., i_u == i, j_u == j_v, j_v == j, k_v == k
            b_w = np.zeros((nm, mp), dtype=float)
            for j in range(m):
                u = i * m + j
                v = j * p + k
                b_w[u, v] = 1.0
            
            # A_w is a matrix of size (nm*mp, r)
            # A_w[idx, t] = UV_t[t, u, v]
            # We can reshape UV_t to (r, nm*mp) and transpose to (nm*mp, r)
            A_w = UV_t[:, :, :].reshape(r, -1).T
            
            # Solve A_w W_w = b_w
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w, :] = sol
        
        return W_sol

    def round_to_half_integer(X):
        X2 = np.round(X * 2)
        return X2 / 2.0

    # Try ranks from 16 up to 30, then 31-49
    # Lower ranks are harder to find but better score.
    ranks_to_try = list(range(16, 31)) + list(range(31, 50))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Increase trials for lower ranks
        max_trials = 2000 if r < 25 else 500
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Sparse random initialization is often better for low rank
            if trial % 2 == 0:
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                for t in range(r):
                    # Pick random non-zero entries
                    nz_u = np.random.choice(nm, size=min(2, nm), replace=False)
                    nz_v = np.random.choice(mp, size=min(2, mp), replace=False)
                    U[nz_u, t] = np.random.choice([-1, 1], size=len(nz_u))
                    V[nz_v, t] = np.random.choice([-1, 1], size=len(nz_v))
            else:
                U = np.random.randn(nm, r) * (1.0 / np.sqrt(r))
                V = np.random.randn(mp, r) * (1.0 / np.sqrt(r))
                U = round_to_half_integer(U)
                V = round_to_half_integer(V)
            
            if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                continue
            
            W_float = solve_for_W(U, V)
            W_r = round_to_half_integer(W_float)
            
            if verify(U, V, W_r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_r.tolist()
                }
    
    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-END
