import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using block structure
    # We suspect rank 32 is achievable via two M_{2,2,5} blocks of rank 16.
    # Let's try to find M_{2,2,5} rank 16 decomposition.
    
    def find_m225_decomposition():
        # Tensor for 2x2 by 2x5
        # A is 2x2 (4 vars), B is 2x5 (10 vars), C is 2x5 (10 vars)
        # T[a_idx, b_idx, c_idx] = 1 if C[c_idx] depends on A[a_idx] and B[b_idx]
        # C is 2x5. Index c in 0..9. c = row*5 + col.
        # A is 2x2. Index a in 0..3. a = row*2 + col.
        # B is 2x5. Index b in 0..9. b = row*5 + col.
        # C[row, col] = sum_{j=0,1} A[row, j] * B[j, col]
        
        T = np.zeros((4, 10, 10))
        for r in range(2):
            for c in range(5):
                c_idx = r * 5 + c
                for j in range(2):
                    a_idx = r * 2 + j
                    b_idx = j * 5 + c
                    T[a_idx, b_idx, c_idx] = 1.0
        
        rank = 16
        # Initialize U, V, W
        np.random.seed(42)
        U = np.random.randn(4, rank)
        V = np.random.randn(10, rank)
        W = np.random.randn(10, rank)
        
        # Define loss function
        def loss(params):
            U_flat = params[:4*rank].reshape(4, rank)
            V_flat = params[4*rank:4*rank+10*rank].reshape(10, rank)
            W_flat = params[4*rank+10*rank:].reshape(10, rank)
            
            # Compute tensor T_est = sum_k U[:,k] outer V[:,k] outer W[:,k]
            # T_est[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
            # This is the CP decomposition.
            # We want to match T.
            
            # Efficient computation:
            # T_est = sum_k np.outer(np.outer(U[:,k], V[:,k]), W[:,k])
            # This is slow for large rank, but rank=16 is small.
            
            T_est = np.zeros((4, 10, 10))
            for k in range(rank):
                T_est += np.einsum('a,b,c->abc', U[:,k], V[:,k], W[:,k])
            
            return np.sum((T - T_est)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use scipy minimize
        try:
            from scipy.optimize import minimize
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8})
            if res.success or res.fun < 1e-6:
                U_final = res.x[:4*rank].reshape(4, rank)
                V_final = res.x[4*rank:4*rank+10*rank].reshape(10, rank)
                W_final = res.x[4*rank+10*rank:].reshape(10, rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_final * 2) / 2
                V_rounded = np.round(V_final * 2) / 2
                W_rounded = np.round(W_final * 2) / 2
                
                # Verify
                T_est = np.zeros((4, 10, 10))
                for k in range(rank):
                    T_est += np.einsum('a,b,c->abc', U_rounded[:,k], V_rounded[:,k], W_rounded[:,k])
                
                if np.allclose(T, T_est, atol=1e-9):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        except Exception:
            pass
        
        return None

    # Try to find the block decomposition
    if time.time() < deadline - 20:
        block_result = find_m225_decomposition()
        if block_result:
            U_b, V_b, W_b = block_result
            rank_block = len(U_b[0])
            total_rank = 2 * rank_block
            
            U_full = [[0] * total_rank for _ in range(n * m)]
            V_full = [[0] * total_rank for _ in range(m * p)]
            W_full = [[0] * total_rank for _ in range(p * n)]
            
            # Block 1: A_1 (rows 0-1 of A, cols 0-1 of A? No, A is 2x4. A_1 is cols 0-1)
            # A_1 is A[:, 0:2]. Indices in U: 0..3.
            # B_1 is B[0:2, :]. Indices in V: 0..9.
            # C is full 2x5. Indices in W: 0..9.
            
            for k in range(rank_block):
                # A_1 part
                for a_idx in range(4):
                    U_full[a_idx][k] = U_b[a_idx][k]
                # B_1 part
                for b_idx in range(10):
                    V_full[b_idx][k] = V_b[b_idx][k]
                # C part
                for c_idx in range(10):
                    W_full[c_idx][k] = W_b[c_idx][k]
                
                # A_2 part
                for a_idx in range(4):
                    U_full[a_idx + 4][k + rank_block] = U_b[a_idx][k]
                # B_2 part
                for b_idx in range(10):
                    V_full[b_idx + 10][k + rank_block] = V_b[b_idx][k]
                # C part
                for c_idx in range(10):
                    W_full[c_idx][k + rank_block] = W_b[c_idx][k]
            
            # Verify full decomposition
            T_full = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_full[i*m+j, j*p+k, k*n+i] = 1.0
                        
            T_est = np.zeros((n*m, m*p, p*n))
            for k in range(total_rank):
                T_est += np.einsum('a,b,c->abc', 
                                    [U_full[i][k] for i in range(n*m)], 
                                    [V_full[i][k] for i in range(m*p)], 
                                    [W_full[i][k] for i in range(p*n)])
            
            if np.allclose(T_full, T_est, atol=1e-9):
                return {"U": U_full, "V": V_full, "W": W_full}
                
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
