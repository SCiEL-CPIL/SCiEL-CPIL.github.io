"""Combined program: Schoolbook baseline with optimized search for lower rank decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 45)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                o = k * n + i
                T[a, b, o] = 1.0
                
    # Helper: verify if U, V, W exactly decompose T
    def verify(U, V, W):
        # Check if sum_c U[:,c] V[:,c]^T W[:,c]^T == T
        # Use einsum for efficiency
        # T[a,b,o] should equal sum_c U[a,c] V[b,c] W[o,c]
        recon = np.einsum('ac,bc,oc->abo', U, V, W)
        # Check if entries are close to integers/half-integers and match T
        if not np.allclose(recon, T, atol=1e-9):
            return False
        # Check half-integer constraint
        scaled = 2 * recon
        if not np.allclose(scaled, np.round(scaled), atol=1e-9):
            return False
        return True
    
    # Helper: round to nearest half-integer
    def round_half(X):
        return np.round(2 * X) / 2.0
    
    # Helper: try to refine a rank R decomposition starting from initial guess
    def try_rank(R, max_iters=500):
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline - 5:
            return False
            
        # Initialize with random small values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # ALS optimization
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U: for each a, U[a,:] = (T[a,:,:] * V * W) / (V * W)
            # T[a,:,:] is (m*p, p*n)
            # We want U[a,c] such that sum_c U[a,c] V[b,c] W[o,c] = T[a,b,o]
            # This is a linear system in U[a,:].
            # Let M be (m*p*p*n, R) where M[(b,o),c] = V[b,c]*W[o,c]
            # Then U[a,:] @ M.T = T[a,:,:].flatten()
            
            # Construct M once (it changes as V,W update, but let's just do it each time for simplicity in first pass)
            # Actually, M depends on V and W.
            # M has shape (m*p*p*n, R)
            # M[idx, c] = V[b,c] * W[o,c] where idx = b*(p*n) + o
            
            M = np.zeros((m*p*p*n, R))
            for b in range(m*p):
                for o in range(p*n):
                    M[b*(p*n)+o, :] = V[b, :] * W[o, :]
                    
            # Solve for U
            # U[a,:] is solution to U[a,:] @ M.T = T[a,:,:].flatten()
            # This is underdetermined if R < m*p*p*n.
            # Use least squares.
            # T_flat has shape (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # U = T_flat @ pinv(M.T)
            # But this is expensive. Let's use normal equations or SVD.
            # Actually, since R is small, we can solve (M M.T) U.T = M T_flat.T
            # M M.T is (m*p*p*n, m*p*p*n) which is large (45*15=675). 675x675 is ok.
            # Wait, m*p*p*n = 3*5*3*5 = 225? No.
            # n=3, m=3, p=5.
            # m*p = 15. p*n = 15.
            # m*p*p*n = 15*15 = 225.
            # So M is 225 x R.
            # M M.T is 225 x 225.
            # This is fast.
            
            try:
                # Solve for U
                # U.T = pinv(M.T) @ T_flat.T
                # Or U = T_flat @ pinv(M.T)
                # Let's use lstsq
                U_new = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
                U = U_new
                
                # Update V
                # V[b,:] solves V[b,:] @ N.T = T[:,b,:].flatten()
                # N[idx, c] = U[a,c] * W[o,c] where idx = a*(p*n) + o
                N = np.zeros((n*m*p*n, R))
                for a in range(n*m):
                    for o in range(p*n):
                        N[a*(p*n)+o, :] = U[a, :] * W[o, :]
                        
                # T[:,b,:].flatten() has shape (n*m, p*n)
                # We need to extract T[a, b, o] for all a, o.
                # T is (n*m, m*p, p*n).
                # T[:, b, :] is (n*m, p*n).
                V_new = np.linalg.lstsq(N.T, T[:, b, :].reshape(n*m*p*n, 1) if False else np.array([T[:, b, o].flatten() for o in range(p*n)]).T, rcond=None)[0].T
                # Wait, the RHS for V[b,:] is T[:, b, :] flattened?
                # Sum_c V[b,c] U[a,c] W[o,c] = T[a,b,o]
                # For fixed b, this is a linear system in V[b,:].
                # LHS is V[b,:] @ (U * W).T
                # Let P be (n*m*p*n, R) where P[(a,o),c] = U[a,c]*W[o,c]
                # Then V[b,:] @ P.T = T[:,b,:].flatten()
                # So V[b,:] = T[:,b,:].flatten() @ pinv(P.T)
                
                # Let's rebuild P correctly
                P = np.zeros((n*m*p*n, R))
                for a in range(n*m):
                    for o in range(p*n):
                        P[a*(p*n)+o, :] = U[a, :] * W[o, :]
                        
                # RHS for each b is T[:, b, :].flatten()
                # We can do this for all b at once?
                # T is (n*m, m*p, p*n).
                # We want RHS[b] = T[:, b, :].flatten()
                # RHS is (m*p, n*m*p*n)
                RHS_V = np.zeros((m*p, n*m*p*n))
                for b in range(m*p):
                    RHS_V[b, :] = T[:, b, :].flatten()
                    
                V_new = np.linalg.lstsq(P.T, RHS_V.T, rcond=None)[0].T
                V = V_new
                
                # Update W
                # W[o,:] solves W[o,:] @ Q.T = T[:,:,o].flatten()
                # Q[idx, c] = U[a,c] * V[b,c] where idx = a*(m*p) + b
                Q = np.zeros((n*m*m*p, R))
                for a in range(n*m):
                    for b in range(m*p):
                        Q[a*(m*p)+b, :] = U[a, :] * V[b, :]
                        
                # RHS for each o is T[:,:,o].flatten()
                RHS_W = np.zeros((p*n, n*m*m*p))
                for o in range(p*n):
                    RHS_W[o, :] = T[:, :, o].flatten()
                    
                W_new = np.linalg.lstsq(Q.T, RHS_W.T, rcond=None)[0].T
                W = W_new
                
            except Exception as e:
                # If linear algebra fails, break
                break
                
        # After ALS, round to half-integers
        U_r = round_half(U)
        V_r = round_half(V)
        W_r = round_half(W)
        
        if verify(U_r, V_r, W_r):
            if R < best_rank:
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
            return True
            
        return False

    # Search for lower ranks
    # Start from 30 and go down to 20?
    # Known lower bounds: For 3x3x3, rank is 23.
    # For 3x3x5, it's likely higher.
    # Let's try a few ranks.
    
    # To save time, we'll only try a few ranks and few iterations.
    ranks_to_try = [35, 30, 25]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            try_rank(R, max_iters=100)
            
    # Convert best result to list of lists for return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
