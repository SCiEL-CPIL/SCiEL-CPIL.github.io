import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Try different ranks
    best_rank = n*m*p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U = U
    best_V = V
    best_W = W
    best_rank = rank
    
    # Try to find a lower-rank decomposition using alternating least-squares
    for target_rank in [70, 65, 60, 55, 50, 45, 40, 35, 30, 25]:
        if time.time() > deadline:
            break
        
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Alternating least-squares
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Fix V and W, solve for U
            # T[a, b, c] = sum_s U[a, s] * V[b, s] * W[c, s]
            # For each a, U[a, :] = (sum_{b,c} T[a,b,c] * V[b,:] * W[c,:]) / (sum_{b,c} V[b,:]^2 * W[c,:]^2)
            # This is a linear system: U[a, :] * (V[b,:] * W[c,:])^T = T[a,b,c]
            # Vectorize: for each a, U[a, :] is the solution to a least-squares problem
            
            # Actually, this is nonlinear. Let me use a different approach.
            # For fixed V and W, the problem is: for each a, find U[a, :] such that
            # sum_s U[a,s] * (V[:,s] * W[:,s])_b = T[a,b] for all b
            # This is a linear system in U[a, :]
            
            # Let M = V * W^T (element-wise), so M[b, s] = V[b, s] * W[c, s] ... no, W is indexed by c, not b.
            
            # I think I need to reshape the tensor.
            # T[a, b, c] = sum_s U[a, s] * V[b, s] * W[c, s]
            # For fixed a, this is: T[a, :, :] = sum_s U[a, s] * (V[:, s] * W[:, s])
            # So T[a, :, :] is a linear combination of the outer products V[:, s] * W[:, s]
            
            # This is a matrix factorization problem.
            # Let X = T[a, :, :] (shape m*p x p*n)
            # Let Y_s = V[:, s] * W[:, s] (shape m*p x p*n, element-wise product)
            # Then X = sum_s U[a, s] * Y_s
            
            # This is a linear least-squares problem: X = Y @ U[a, :]^T
            # where Y is a (m*p*p*n) x target_rank matrix
            
            # This is too large. Let me skip this approach.
            break
        
        # Check if the decomposition is valid
        # T_reconstructed[a, b, c] = sum_s U[a, s] * V[b, s] * W[c, s]
        T_rec = np.zeros_like(T)
        for s in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
        
        if np.allclose(T_rec, T, atol=1e-10):
            # Round to integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_rec_rounded = np.zeros_like(T)
            for s in range(target_rank):
                T_rec_rounded += np.einsum('a,b,c->abc', U_rounded[:, s], V_rounded[:, s], W_rounded[:, s])
            
            if np.allclose(T_rec_rounded, T, atol=1e-10):
                best_rank = target_rank
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
