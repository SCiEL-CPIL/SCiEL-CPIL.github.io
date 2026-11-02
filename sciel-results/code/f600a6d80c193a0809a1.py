"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a = i*m+j (A index), b = j'*p+k (B index), c = k'*n+i' (C index)
    # T[a,b,c] = 1 iff j=j' and k=k' and i=i'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Greedy rank-1 peeling
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    T_work = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find the maximum entry
        max_idx = np.unravel_index(np.argmax(np.abs(T_work)), T_work.shape)
        max_val = T_work[max_idx]
        
        if abs(max_val) < 1e-10:
            break
        
        # Extract rank-1 component
        a, b, c = max_idx
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        # For matrix multiplication tensor, rank-1 components have special structure
        # Try to extract a rank-1 that matches the tensor structure
        # A rank-1 for matmul: u[i*m+j] * v[j'*p+k] * w[k'*n+i']
        # This should be nonzero when j=j', k=k', i=i'
        
        # Simple approach: extract based on the max position
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        
        # Build rank-1 vectors for this (i,j,k) position
        u_vec[i*m+j] = 1.0
        v_vec[j*p+k] = 1.0
        w_vec[k*n+i] = 1.0
        
        # Subtract this rank-1 from T_work
        T_work -= np.outer(np.outer(u_vec, v_vec), w_vec).reshape(n*m, m*p, p*n)
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    rank_found = len(U_cols)
    
    if rank_found < best_rank:
        best_rank = rank_found
        best_U = np.array(U_cols).T
        best_V = np.array(V_cols).T
        best_W = np.array(W_cols).T
    
    # Convert to half-integers (they should already be 0 or 1)
    if best_U is not None:
        U_out = [[int(round(x * 2)) / 2 for x in row] for row in best_U]
        V_out = [[int(round(x * 2)) / 2 for x in row] for row in best_V]
        W_out = [[int(round(x * 2)) / 2 for x in row] for row in best_W]
    else:
        # Fallback to schoolbook
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
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
