import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank for 3x3x3 is 20. For 3x3x5, we can likely use 20.
    # We will try to find a decomposition of rank 20.
    
    def target_tensor():
        # M_{i, j, j', k, i', k'} = 1 if i==i', j==j', k==k' else 0
        # This is a 6D tensor, but we can flatten it.
        # Actually, we just need to check the conditions.
        pass

    # We will use a simple iterative method.
    # Initialize U, V, W with random values.
    # Minimize || sum_t U_t V_t W_t - M ||^2
    
    # For 3x3x3, the tensor size is 3x3x3.
    # For 3x3x5, the tensor size is 3x3x5.
    # We can solve for 3x3x3 and then extend to 3x3x5 by repeating the columns.
    
    # Let's solve for 3x3x3 first.
    n3, m3, p3 = 3, 3, 3
    rank = 20
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n3 * m3, rank)
    V = np.random.randn(m3 * p3, rank)
    W = np.random.randn(p3 * n3, rank)
    
    # Target tensor T_{i, j, k} = 1
    # Condition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # And sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t] = 0 if (j'!=j or k'!=k or i'!=i)
    
    # We can define the error as the sum of squares of the violations.
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    best_U, best_V, best_W = U, V, W
    best_err = float('inf')
    
    lr = 0.01
    
    for it in range(1000):
        if time.time() > deadline:
            break
            
        # Compute gradient
        # This is complex. We will use a simple heuristic:
        # Randomly perturb and keep if error decreases.
        
        # Calculate error
        err = 0.0
        # Check diagonal terms
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m3+j, t] * V[j*p3+k, t] * W[k*n3+i, t]
                    err += (val - 1.0)**2
        # Check off-diagonal terms (sample a few)
        # To save time, we check a subset of off-diagonal terms.
        for _ in range(100):
            i = np.random.randint(n3)
            j = np.random.randint(m3)
            k = np.random.randint(p3)
            i2 = np.random.randint(n3)
            j2 = np.random.randint(m3)
            k2 = np.random.randint(p3)
            if i==i2 and j==j2 and k==k2:
                continue
            val = 0.0
            for t in range(rank):
                val += U[i*m3+j, t] * V[j2*p3+k2, t] * W[k2*n3+i2, t]
            err += val**2
            
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-6:
            break
            
        # Update
        # Simple random walk
        U += np.random.randn(n3 * m3, rank) * lr
        V += np.random.randn(m3 * p3, rank) * lr
        W += np.random.randn(p3 * n3, rank) * lr
        
    # Extend to 3x3x5
    # U is n*m x rank = 3*3 x 20
    # V is m*p x rank = 3*5 x 20
    # W is p*n x rank = 5*3 x 20
    
    U_final = np.zeros((n*m, rank))
    V_final = np.zeros((m*p, rank))
    W_final = np.zeros((p*n, rank))
    
    # Copy U
    U_final[:n3*m3, :] = best_U
    
    # Copy V and W
    # For k=0,1,2, use the 3x3 solution
    # For k=3,4, reuse k=0,1
    for k in range(p):
        k3 = k % 3
        # V rows are j*p + k
        for j in range(m3):
            V_final[j*p+k, :] = best_V[j*p3+k3, :]
        # W rows are k*n + i
        for i in range(n3):
            W_final[k*n+i, :] = best_W[k3*n3+i, :]
            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

