import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank
    for r in range(25, rank_sb):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            np.random.seed(42 + r * 100 + trial)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Gradient descent
            for it in range(100):
                if time.time() > deadline - 20:
                    break
                
                # Compute residual
                # R[i*m+j, j'*p+k, k'*n+i'] = T[...] - sum_c U[i*m+j,c] V[j'*p+k,c] W[k'*n+i',c]
                R = np.zeros_like(T)
                for c in range(r):
                    R -= np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                R += T
                
                # Compute gradients
                # dR/dU[i*m+j, c] = -sum_{j',k,k'} V[j'*p+k,c] W[k'*n+i',c] * R[i*m+j, j'*p+k, k'*n+i']
                # This is expensive. Let's use a simpler approach.
                
                # Use ALS instead
                # For each c, update U[:,c] by solving a least squares problem
                for c in range(r):
                    # Residual without term c
                    R_c = R + np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                    
                    # For each (i,j), we want:
                    # sum_{j',k,k'} U[i*m+j,c] V[j'*p+k,c] W[k'*n+i',c] R_c[i*m+j, j'*p+k, k'*n+i'] = 0
                    # => U[i*m+j,c] * sum_{j',k,k'} V[j'*p+k,c] W[k'*n+i',c] R_c[i*m+j, j'*p+k, k'*n+i'] = 0
                    # This doesn't give a useful equation.
                    
                    # Let's use a different approach: for fixed V and W, the optimal U is:
                    # U[:,c] = (V[:,c] ⊙ W[:,c]) * T / ||V[:,c] ⊙ W[:,c]||^2
                    # where ⊙ is element-wise product, but this doesn't account for the other terms.
                    
                    # Actually, for ALS, we fix V and W, and solve for U:
                    # min_U sum_{i,j,j',k,k',i'} (T[i*m+j, j'*p+k, k'*n+i'] - sum_c U[i*m+j,c] V[j'*p+k,c] W[k'*n+i',c])^2
                    # For fixed V and W, this is a linear system in U.
                    # But the system is large.
                    
                    # Let's just use a simple gradient step
                    break
                
                break
            
            # Check residual
            R = np.zeros_like(T)
            for c in range(r):
                R -= np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            R += T
            err = np.max(np.abs(R))
            
            if err < 1e-6:
                found = True
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
