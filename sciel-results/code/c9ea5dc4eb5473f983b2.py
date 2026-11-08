import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The multiplication tensor: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # We want to minimize r (rank)
    
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p
    best_decomp = None
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 16, 8) = 16
    # But actually the tensor rank lower bound is more complex
    # Let's try from 16 up to 32
    
    # For each rank, try random searches
    for rank in range(16, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 2000 if rank <= 20 else 500
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Initialize U, V, W with random half-integers
            # Values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(nm)])
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(mp)])
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(pn)])
            
            # Check if this satisfies all constraints
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_decomp = (U, V, W)
                break  # Found a valid decomposition for this rank
    
    # If no better decomposition found, use schoolbook
    if best_decomp is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    else:
        U, V, W = best_decomp
        # Convert to list of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
