import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
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
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try to find lower rank using structured random search
    # Key insight: the tensor has a lot of symmetry, so we can use that
    
    # For each target rank, try multiple random initializations
    for target_rank in range(100, 90, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + target_rank)
            
            # Initialize with small random values
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            
            # Alternating least squares style refinement
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update W: for each (k,i), we want
                # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for each j
                # This is a system of m equations for W[k*n+i,:]
                # Let A_{j,c} = U[i*m+j,c] * V[j*p+k,c]
                # Then A * W[k*n+i,:] = ones(m)
                
                # Solve least squares
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                        # Solve A * x = ones(m)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i,:] = x
                        except:
                            pass
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j,:] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k,:] = x
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
                            err += (val - 1.0)**2
                if err < 1e-4:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j,:] * V_r[j*p+k,:] * W_r[k*n+i,:])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        found = True
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
