import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    def compute_error(U, V, W, r):
        # Vectorized error computation
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Target: T[i*m+j, j'*p+k] = 1 if j==j' else 0
        # W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c]
        
        # Reshape for broadcasting
        # U_reshaped: (n, m, r)
        U_r = U.reshape(n, m, r)
        # V_reshaped: (m, p, r)
        V_r = V.reshape(m, p, r)
        # W_reshaped: (p, n, r)
        W_r = W.reshape(p, n, r)
        
        # Compute sum over c: W[k,n,i] * U[i,m,j] * V[j',p,k]
        # Result shape: (n, m, m, p)
        # Indices: i, j, j', k
        
        # U[i, j, c] * V[j', k, c] -> (n, m, m, p, r)
        # This is too large.
        
        # Alternative:
        # For each c, compute outer products?
        # Sum over c of W_c * (U_c * V_c)
        
        # Let's just loop over c if r is small.
        max_err = 0.0
        for c in range(r):
            u_c = U[:, c] # (n*m)
            v_c = V[:, c] # (m*p)
            w_c = W[:, c] # (p*n)
            
            # Expand u_c to (n, m)
            u_2d = u_c.reshape(n, m)
            # Expand v_c to (m, p)
            v_2d = v_c.reshape(m, p)
            # Expand w_c to (p, n)
            w_2d = w_c.reshape(p, n)
            
            # Product: (n, m) * (m, p) -> (n, p) ? No.
            # We need T[i, j, j', k] = W[k, i] * U[i, j] * V[j', k]
            
            # Term: U[i, j] * V[j', k] * W[k, i]
            # This is a 4D tensor.
            
            # We can compute the full tensor for this c and accumulate.
            # Tensor_c[i, j, j', k] = U[i, j] * V[j', k] * W[k, i]
            
            # Use broadcasting
            # U: (n, 1, 1, m) -> (n, m, 1, 1) ? No.
            # U[i, j] -> (n, m, 1, 1)
            u_exp = u_2d[:, :, np.newaxis, np.newaxis] # (n, m, 1, 1)
            # V[j', k] -> (1, 1, m, p)
            v_exp = v_2d[np.newaxis, np.newaxis, :, :] # (1, 1, m, p)
            # W[k, i] -> (1, 1, 1, p) * (n, 1, 1, 1) ?
            # W[k, i] is (p, n). We need (n, 1, 1, p).
            w_exp = w_2d.T[np.newaxis, np.newaxis, :, :] # (1, 1, n, p) -> wait.
            # w_2d is (p, n). w_2d.T is (n, p).
            # We want index i for U, k for V/W.
            # Tensor[i, j, j', k] = U[i, j] * V[j', k] * W[k, i]
            # U: (n, m) -> (n, m, 1, 1)
            # V: (m, p) -> (1, 1, m, p)
            # W: (p, n) -> (n, 1, 1, p)
            
            w_exp = w_2d.T[:, np.newaxis, np.newaxis, :] # (n, 1, 1, p)
            
            tensor_c = u_exp * v_exp * w_exp # (n, m, m, p)
            
            # Target: 1 if j == j' else 0.
            # Target[i, j, j', k] = 1 if j == j' else 0.
            # This is independent of i, k.
            
            # We can check error for this c? No, sum over c.
            # We need to sum tensor_c over c.
            
            if c == 0:
                total_tensor = np.zeros((n, m, m, p))
            total_tensor += tensor_c
            
        # Now check error
        # Target is 1 if j == j'
        target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i, j, j, k] = 1.0
                    
        err = np.max(np.abs(total_tensor - target))
        return err

    def try_rank(r):
        max_trials = 100
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(99999 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            err = compute_error(U, V, W, r)
            if err < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if compute_error(U_r, V_r, W_r, r) < 1e-6:
                    return (U_r, V_r, W_r)
        
        return None

    # Schoolbook fallback
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
    
    # Try to find lower rank
    for r in range(36, rank):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return {"U": U, "V": V, "W": W}
