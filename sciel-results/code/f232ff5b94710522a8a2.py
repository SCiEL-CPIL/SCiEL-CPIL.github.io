import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin

    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[a, b, c]
    T = np.zeros((nm, mp, pn))
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                nonzero_entries.append((a, b, c))

    def verify(U, V, W, tol=1e-9):
        # Check if U, V, W are close to half-integers
        # And if they exactly decompose T
        # The problem says "integer/half-integer entries"
        # So we check if rounding to 0.5 gives an exact match
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if U, V, W are close to U_r, V_r, W_r
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Check exact decomposition with rounded values
        # sum_r U_r[a,r] V_r[b,r] W_r[c,r] == T[a,b,c]
        
        # Efficient check
        # For each a, compute M_a[b, c] = sum_r U_r[a,r] V_r[b,r] W_r[c,r]
        # This is U_r[a,:] dot (V_r * W_r.T) ? No.
        # M_a = U_r[a, :] @ (V_r * W_r.T) is not right dimensions.
        # Let's just loop for simplicity as nm, mp, pn are small.
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                    if abs(val - T[a, b, c]) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
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

    fallback = schoolbook()

    def try_als(rank, max_iters=100, num_restarts=3):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Pre-compute indices for speed?
            # We need to solve for each row of U, V, W.
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(nm):
                    # Construct A: shape (mp*pn, rank)
                    # A[idx, r] = V[b, r] * W[c, r] where idx = b*pn + c
                    # t: shape (mp*pn,)
                    # t[idx] = T[a, b, c]
                    
                    # Vectorized construction of A
                    # V is (mp, rank), W is (pn, rank)
                    # We want outer product of V rows and W rows element-wise?
                    # No, A[b*pn+c, r] = V[b, r] * W[c, r]
                    # This is a matmul if we reshape?
                    # Let's just use broadcasting.
                    # V[:, None, :] * W[None, :, :] -> (mp, pn, rank)
                    # Reshape to (mp*pn, rank)
                    A = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    t = T[a, :, :].reshape(-1)
                    
                    # Solve A U_a^T = t
                    # Use lstsq
                    U[a, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                
                # Update V
                for b in range(mp):
                    # A[idx, r] = U[a, r] * W[c, r] where idx = a*pn + c
                    A = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    t = T[:, b, :].reshape(-1)
                    V[b, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                    
                # Update W
                for c in range(pn):
                    # A[idx, r] = U[a, r] * V[b, r] where idx = a*mp + b
                    A = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                    t = T[:, :, c].reshape(-1)
                    W[c, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                    
                # Check convergence / validity
                if it % 5 == 0:
                    if verify(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Try ranks from low to high
    # Lower bound for 4x5x6? 
    # Rank 49 is for 4x4x4.
    # Let's try from 60 to 119.
    best_result = None
    for r in range(60, 120):
        if time.time() > deadline - 5:
            break
            
        result = try_als(r, max_iters=50, num_restarts=2)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = fallback
        
    return best_result
