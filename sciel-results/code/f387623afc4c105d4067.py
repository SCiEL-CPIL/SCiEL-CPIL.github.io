import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    # Convert to list format for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    deadline = time.time() + 130  # 130s safety margin
    
    best_rank = rank_fb
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try ranks from lower to higher
    # Theoretical lower bound for matmul (n,m,p) is at least max(n*m, m*p, p*n)
    # For (3,4,7): max(12, 28, 21) = 28
    # Known: rank of matmul(3,4,7) - let's search from 28 upward
    
    min_rank = max(n * m, m * p, p * n)  # 28
    
    # Try various ranks
    ranks_to_try = list(range(min_rank, min(rank_fb + 1, min_rank + 30)))
    random.shuffle(ranks_to_try)
    ranks_to_try.sort()  # Try lower ranks first
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Multiple random attempts for this rank
        max_attempts = 50 if target_rank <= 35 else 20
        if target_rank >= 40:
            max_attempts = 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterative refinement: minimize sum of squared errors
            # T[i,j,k] should be 1 for all (i,j,k)
            # Residual[i,j,k] = 1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Use gradient-free approach: random perturbations
            # Or use a simple iterative method
            
            # Actually, let's use a different approach:
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of 84 bilinear equations in 61*r variables
            
            # Try using scipy's least_squares or similar
            # But the problem is bilinear, not linear
            
            # Alternative: fix two factors, solve for third (linear)
            # Then iterate
            
            # Let's try: fix V and W, solve for U (linear system)
            # For each row i*m+j of U, we need:
            # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
            # This is a linear system: U[i*m+j, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1
            
            # Actually for fixed j, i:
            # For each k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Let a_k,c = V[j*p+k, c] * W[k*n+i, c]
            # Then sum_c U[i*m+j, c] * a_k,c = 1 for k=0..p-1
            # This is p equations in r unknowns for each (i,j)
            
            # Total: n*m*p = 84 equations, n*m*r = 12r unknowns for U
            # For r >= 7 (p=7), we can solve
            
            # Iterate: fix V,W -> solve U; fix U,W -> solve V; fix U,V -> solve W
            
            success = False
            for iteration in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Solve for U given V, W
                U_new = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # For each k, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # Build system: A @ u = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        # Solve least squares
                        if target_rank >= p:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        else:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i * m + j, :] = u
                
                # Solve for V given U, W
                V_new = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        # For each i, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # A[i,c] = U[i*m+j,c] * W[k*n+i,c], b[i] = 1
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j * p + k, :] = v
                
                # Solve for W given U, V
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        # For each j, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # A[j,c] = U[i*m+j,c] * V[j*p+k,c], b[j] = 1
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k * n + i, :] = w
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    success = True
                    break
                
                # Damping to avoid oscillation
                # (already overwrote, so skip damping for simplicity)
            
            if success:
                # Verify exactly
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-9:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Found a valid decomposition for this rank, try lower
        
        if time.time() > deadline - 10:
            break
    
    # Convert best to list format
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
