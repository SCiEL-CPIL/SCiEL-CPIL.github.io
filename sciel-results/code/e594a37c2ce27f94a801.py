import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Pre-compute indices for vectorized error calculation
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    
    # Target tensor T[i, j, l, k] = delta(j, l)
    # We compute S[i, j, l, k] = sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r]
    
    def compute_error_np(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to tensors
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1) # V index l*p+k -> V_t[l, k, r]
        W_t = W.reshape(p, n, -1) # W index k*n+i -> W_t[k, i, r]
        
        # Compute sum over r
        # S[i, j, l, k] = sum_r U_t[i, j, r] * V_t[l, k, r] * W_t[k, i, r]
        
        # This is a contraction.
        # Let's use einsum or explicit loops if R is small?
        # R is small (16-24).
        
        # S = np.einsum('ijr,lkr,ki r->ijlk', U_t, V_t, W_t) ? No, indices don't match directly for einsum easily without broadcasting tricks.
        # Let's do it with loops over r, but vectorized over i,j,l,k.
        
        S = np.zeros((n, m, m, p), dtype=float)
        for r in range(U.shape[1]):
            # U[:, :, r] is (n, m)
            # V[:, :, r] is (m, p)
            # W[:, :, r] is (p, n)
            
            # Term_r[i, j, l, k] = U[i, j] * V[l, k] * W[k, i]
            # U[i,j] * W[k,i] -> (n, m, p, n) ? No.
            # Let's broadcast:
            # U_part = U_t[:, :, r] # (n, m)
            # V_part = V_t[:, :, r] # (m, p)
            # W_part = W_t[:, :, r] # (p, n)
            
            # We want S[i, j, l, k] += U[i, j] * V[l, k] * W[k, i]
            # U[i, j] * W[k, i] -> shape (n, m, p, n) is wrong.
            # U[i, j] * W[i, k] (if W was (n,p))? W is (p,n). W[k, i].
            # So U[i, j] * W[k, i] is (n, m, p, n) -> indices i, j, k, i.
            # This collapses i? No, i is in U and W.
            # Let's use einsum for the product of U and W first?
            # P[i, j, k] = sum_r? No, we are inside r loop.
            
            # Let's just do the full einsum for each r?
            # S_r = np.einsum('ij,lk,ki->ijlk', U_t[:,:,r], V_t[:,:,r], W_t[:,:,r])
            S += np.einsum('ij,lk,ki->ijlk', U_t[:, :, r], V_t[:, :, r], W_t[:, :, r])
            
        # Target
        # T[i, j, l, k] = 1 if j==l else 0
        # Note: T is independent of i, k.
        target = np.zeros((n, m, m, p), dtype=float)
        for j in range(m):
            target[:, j, j, :] = 1.0
            
        diff = S - target
        return np.sum(diff * diff)

    def local_search_np(U, V, W, R, time_limit):
        start_time = time.time()
        current_error = compute_error_np(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        while time.time() - start_time < time_limit:
            # Pick random variable
            if random.random() < 0.3:
                mat = U
                rows = n * m
            elif random.random() < 0.5:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, R - 1)
            
            old_val = mat[r_idx, c_idx]
            new_val = old_val + random.choice([-1, 1])
            
            mat[r_idx, c_idx] = new_val
            new_error = compute_error_np(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                mat[r_idx, c_idx] = old_val
            
            if current_error < 1e-9:
                return U, V, W
        
        return best_U, best_V, best_W

    def try_find_decomposition(R, time_limit=10):
        start_time = time.time()
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)], dtype=float)
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)], dtype=float)
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p * n)], dtype=float)
            
            U_res, V_res, W_res = local_search_np(U, V, W, R, time_limit=2)
            
            if compute_error_np(U_res, V_res, W_res) < 1e-9:
                # Convert to list of lists of ints (or half-ints if needed, but we used ints)
                return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}
        
        return None

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    for R in range(16, 24):
        result = try_find_decomposition(R, time_limit=10)
        if result:
            return result

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
