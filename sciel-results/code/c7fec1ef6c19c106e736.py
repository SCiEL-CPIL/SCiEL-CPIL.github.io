"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s limit, return trivial if we run out of time
    def time_exceeded():
        return (time.time() - start_time) > 140
    
    # Trivial rank-45 decomposition
    def get_trivial():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verification function: check if U, V, W with given rank form a valid
    # decomposition of the matrix multiplication tensor.
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # The multiplication tensor has T[a,b,c] = 1 iff (i,j,k) are consistent
    # (i.e., a corresponds to (i,j), b to (j,k), c to (k,i)) and 0 otherwise.
    def verify_decomposition(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all "active" positions: (i, j, k) for i in range(n), j in range(m), k in range(p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        
        # Check a sample of "inactive" positions to ensure no cross-talk
        # Total positions: (n*m) * (m*p) * (p*n) = 9*15*15 = 2025
        # Active positions: n*m*p = 45
        # We'll check all inactive positions for correctness
        active_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    active_set.add((a, b, c))
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in active_set:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using a randomized algebraic approach.
    # Strategy: The matrix multiplication tensor for 3x3 * 3x5 can potentially be
    # decomposed at a rank lower than 45. We attempt a structured search.
    
    # Approach: Try rank values from 44 down to 40, using a randomized 
    # initialization and gradient-free refinement.
    
    # For each candidate rank r, we try to find U (9 x r), V (15 x r), W (15 x r)
    # such that the tensor identity holds.
    
    # We use a simple approach: fix the structure and solve for coefficients.
    # The tensor equation is:
    #   For all (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a system of 2025 equations in 9r + 15r + 15r = 39r unknowns.
    # For r=45: 1755 unknowns, 2025 equations (overdetermined but trivial solution exists)
    # For r=44: 1716 unknowns, 2025 equations (still overdetermined)
    
    # A more practical approach: use the fact that the multiplication tensor
    # has a specific structure. We can try to find a decomposition by:
    # 1. Starting with the trivial 45-term decomposition
    # 2. Trying to merge or cancel terms
    
    # Alternative: Use known results. For 3x3x3, Strassen gives rank 7.
    # For 3x3x5, we can potentially use a block approach.
    
    # Let's try a direct approach: assume a decomposition of rank r where
    # the U, V, W matrices have a specific sparse structure.
    
    # Actually, let's try a simpler heuristic: search for rank 44 by 
    # finding two terms in the trivial decomposition that can be combined.
    
    # In the trivial decomposition, each term r corresponds to (i_r, j_r, k_r).
    # Term r: U[e_r, r]=1, V[f_r, r]=1, W[g_r, r]=1
    # where e_r = i_r*m + j_r, f_r = j_r*p + k_r, g_r = k_r*n + i_r
    
    # Two terms r1 and r2 can potentially be replaced by two different terms
    # that span the same subspace. This is complex.
    
    # Let's try a different approach: use randomized SVD-like methods on the
    # tensor to find a low-rank approximation, then round to integers/half-integers.
    
    # For each candidate rank r from 44 down to 40:
    for r in range(44, 39, -1):
        if time_exceeded():
            break
        
        # Initialize with random values
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # We need to solve: for all (a,b,c), sum_r' U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
        # This is a nonlinear system. We can use a simple iterative approach.
        
        # Actually, this is very hard to solve directly. Let's try a different strategy.
        
        # Strategy: Use the fact that the tensor is bilinear in each factor.
        # Fix V and W, solve for U (linear system), then fix U and W, solve for V, etc.
        
        # Initialize with trivial decomposition projected to rank r
        U_triv = np.zeros((n * m, n * m * p))
        V_triv = np.zeros((m * p, n * m * p))
        W_triv = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_triv[i * m + j, col] = 1
                    V_triv[j * p + k, col] = 1
                    W_triv[k * n + i, col] = 1
                    col += 1
        
        # Project to rank r using SVD
        # This won't directly give us a valid decomposition, but it's a starting point.
        
        # Let's try a more direct approach: use the alternating least squares method
        # for tensor decomposition.
        
        # Target tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # ALS: alternate between solving for U, V, W given the other two
        # For U: for each a, U[a, :] should satisfy sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # for all (b,c). This is: U[a, :] * (V[:, b] * W[:, c]) = T[a, b, c]
        # Let M_a[b, r] = sum_c' V[b, r] W[c', r] ... no, this is not quite right.
        
        # Actually, for fixed V and W, the equation for U[a, :] is:
        # sum_r U[a,r] * (sum over... no, it's for each (b,c) pair)
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Let X[b, r] = V[b, r] and Y[c, r] = W[c, r]
        # Then T[a,b,c] = sum_r U[a,r] X[b,r] Y[c,r]
        # This means: for each a, the matrix T[a, :, :] = sum_r U[a,r] X[:, r] Y[:, r]^T
        # So T[a, :, :] = X diag(U[a, :]) Y^T
        # This is a rank-r decomposition of T[a, :, :]!
        
        # So we can solve: for each a, find U[a, :] such that T[a,:,:] ≈ X diag(U[a,:]) Y^T
        # where X = V, Y = W.
        
        # Given V and W, for each a:
        # T[a,:,:] = V diag(u_a) W^T
        # where u_a = U[a, :]
        # This is: T[a,:,:] = V * diag(u_a) * W^T
        # Multiply by V^T on left and W on right:
        # V^T T[a,:,:] W = V^T V diag(u_a) W^T W
        # Let G = V^T V (r x r), H = W^T W (r x r)
        # Let S = V^T T[a,:,:] W (r x r)
        # Then S = G diag(u_a) H
        # So diag(u_a) = G^{-1} S H^{-1} (if G and H are invertible)
        # u_a = diagonal of G^{-1} S H^{-1}
        
        # Similarly for V and W.
        
        # Let's implement this ALS approach.
        
        # Initialize
        U_als = np.zeros((n * m, r))
        V_als = np.zeros((m * p, r))
        W_als = np.zeros((p * n, r))
        
        # Initialize with random values
        U_als = np.random.randn(n * m, r) * 0.1 + 0.5
        V_als = np.random.randn(m * p, r) * 0.1 + 0.5
        W_als = np.random.randn(p * n, r) * 0.1 + 0.5
        
        # ALS iterations
        max_iters = 50
        for it in range(max_iters):
            if time_exceeded():
                break
            
            # Update U
            # For each a: S = V^T T[a,:,:] W, then u_a = diag(G^{-1} S H^{-1})
            G = V_als.T @ V_als  # r x r
            H = W_als.T @ W_als  # r x r
            
            # Check if G and H are invertible
            try:
                G_inv = np.linalg.inv(G + 1e-10 * np.eye(r))
                H_inv = np.linalg.inv(H + 1e-10 * np.eye(r))
            except:
                break
            
            for a in range(n * m):
                S = V_als.T @ T[a, :, :] @ W_als  # r x r
                M = G_inv @ S @ H_inv
                U_als[a, :] = np.diag(M)
            
            # Update V
            # For each b: T[:, b, :] = U diag(v_b) W^T
            # U^T T[:, b, :] W = U^T U diag(v_b) W^T W
            # Let P = U^T U, Q = W^T W
            # S_v = U^T T[:, b, :] W
            # v_b = diag(P^{-1} S_v Q^{-1})
            P = U_als.T @ U_als
            Q = W_als.T @ W_als
            try:
                P_inv = np.linalg.inv(P + 1e-10 * np.eye(r))
                Q_inv = np.linalg.inv(Q + 1e-10 * np.eye(r))
            except:
                break
            
            for b in range(m * p):
                S_v = U_als.T @ T[:, b, :] @ W_als
                M_v = P_inv @ S_v @ Q_inv
                V_als[b, :] = np.diag(M_v)
            
            # Update W
            # For each c: T[:, :, c] = U diag(w_c) V^T
            # U^T T[:, :, c] V = U^T U diag(w_c) V^T V
            # S_w = U^T T[:, :, c] V
            # w_c = diag(P^{-1} S_w G^{-1})
            R = V_als.T @ V_als
            try:
                P_inv = np.linalg.inv(P + 1e-10 * np.eye(r))
                R_inv = np.linalg.inv(R + 1e-10 * np.eye(r))
            except:
                break
            
            for c in range(p * n):
                S_w = U_als.T @ T[:, :, c] @ V_als
                M_w = P_inv @ S_w @ R_inv
                W_als[c, :] = np.diag(M_w)
        
        # Check if the solution is valid
        if not time_exceeded():
            # Convert to lists and verify
            U_list = U_als.tolist()
            V_list = V_als.tolist()
            W_list = W_als.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no lower-rank decomposition found, return trivial
    return get_trivial()
# EVOLVE-BLOCK-END
