"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition of T."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is equivalent to: for each c, the matrix T[:,:,c] = U * diag(W[:,c]) * V.T
        # Or: T reshaped as (n*m, m*p*p*n) ... let's just check directly
        # More efficient: for each r, add outer(U[:,r], outer(V[:,r], W[:,r]))
        # But that's O(R * n*m * m*p * p*n) which is too slow
        # Instead, check: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] W[c,r])
        # Let M_r = V[:,r] * W[:,r].T (outer product), shape (m*p, p*n)
        # Then T[a,:,:] = sum_r U[a,r] * M_r
        # So T reshaped as (n*m, m*p, p*n) should equal sum_r U[:,r] * (V[:,r] * W[:,r].T)
        # Check: T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r] should be ~0
        
        # Efficient check: compute the difference using matrix operations
        # For each r, the contribution is U[:,r] * V[:,r].T * W[:,r] ... no
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Let's compute: for each a, T[a,:,:] should equal sum_r U[a,r] * (V[:,r] * W[:,r].T)
        
        diff = np.zeros_like(T)
        for r in range(U.shape[1]):
            diff += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        
        return np.max(np.abs(diff - T)) < tol
    
    # Try ranks from 55 down to 50
    for R in range(55, 49, -1):
        if time.time() > deadline:
            break
        # Try a few random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random initialization
            U_rand = np.random.randn(n*m, R)
            V_rand = np.random.randn(m*p, R)
            W_rand = np.random.randn(p*n, R)
            
            # Simple verification (might be too slow for large R)
            # Instead, use a faster check
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Check a few random (a,b,c) triples
            # Better: check the full thing but efficiently
            # T = sum_r U[:,r] * (V[:,r] * W[:,r].T) ... no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is a 3-way tensor, can check by:
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r].T)
            
            # Let's just do a quick check on a subset
            # Actually, for small tensors, just do the full check
            # But R*n*m*m*p*p*n might be too large... 55*4*30*30 = 198000, that's fine
            
            # Compute the difference
            # T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
            # = T[a,b,c] - (U * (V.T * W.T.T))... 
            # Let's compute: for each r, outer(U[:,r], V[:,r]) gives (n*m, m*p) matrix
            # Then multiply element-wise with W[:,r] and sum... 
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # = sum_r (U[a,r] V[b,r]) W[c,r]
            # Let M[a,b] = sum_r U[a,r] V[b,r] ... no, that's not right
            # M_r[a,b] = U[a,r] * V[b,r], then T[a,b,c] = sum_r M_r[a,b] * W[c,r]
            # So T[:,:,c] = sum_r (U[:,r] * V[:,r]) * W[c,r] ... 
            # = sum_r outer(U[:,r], V[:,r]) * W[c,r]
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
            # where U[:,r] * V[:,r] is element-wise product, shape (n*m, m*p)
            # So T[:,:,c] = sum_r W[c,r] * (U * V)[:,:,r]... 
            # Let UV = U * V (element-wise, shape (n*m, m*p, R))
            # Then T[a,b,c] = sum_r UV[a,b,r] * W[c,r]
            # So T reshaped as (n*m, m*p, p*n) = UV @ W.T ... 
            # UV shape: (n*m, m*p, R), W.T shape: (R, p*n)
            # (UV @ W.T)[a,b,c] = sum_r UV[a,b,r] * W.T[r,c] = sum_r UV[a,b,r] * W[c,r] ✓
            
            UV = U_rand * V_rand[None, :, :]  # (n*m, m*p, R)
            T_check = UV @ W_rand.T  # (n*m, m*p, p*n)
            
            if np.max(np.abs(T_check - T)) < 1e-8:
                best_rank = R
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break
        if best_rank < rank_schoolbook:
            break
    
    if best_rank < rank_schoolbook:
        # Convert to lists
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END

