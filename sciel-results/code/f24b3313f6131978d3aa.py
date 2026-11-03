<<<<<<< SEARCH
"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
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

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
=======
"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build the target tensor
    shape = (n*m, m*p, p*n)
    T = np.zeros(shape)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def als_decompose(T, rank, max_iters=40, tol=1e-8, seed=0):
        """Alternating Least Squares for tensor decomposition"""
        rng = np.random.RandomState(seed)
        s1, s2, s3 = T.shape
        U = rng.randn(s1, rank)
        V = rng.randn(s2, rank)
        W = rng.randn(s3, rank)
        err = np.inf
        
        for it in range(max_iters):
            # Update U
            for i in range(s1):
                A_i = np.array([(V[:,r] * W[:,r]).flatten() for r in range(rank)]).T
                rhs = T[i,:,:].flatten()
                U[i,:] = np.linalg.lstsq(A_i, rhs, rcond=None)[0]
            
            # Update V
            for j in range(s2):
                A_j = np.array([(U[:,r] * W[:,r]).flatten() for r in range(rank)]).T
                rhs = T[:,j,:].flatten()
                V[j,:] = np.linalg.lstsq(A_j, rhs, rcond=None)[0]
            
            # Update W
            for k in range(s3):
                A_k = np.array([(U[:,r] * V[:,r]).flatten() for r in range(rank)]).T
                rhs = T[:,:,k].flatten()
                W[k,:] = np.linalg.lstsq(A_k, rhs, rcond=None)[0]
            
            # Check convergence
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T - T_rec))
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try to find lower rank decompositions
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from 59 down to 55
    for rank in range(59, 54, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(3):
            if time.time() > deadline:
                break
            
            U, V, W, err = als_decompose(T, rank, max_iters=30, tol=1e-7, seed=seed)
            if err < 1e-6:
                found = True
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to trivial decomposition
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

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
>>>>>>> REPLACE
