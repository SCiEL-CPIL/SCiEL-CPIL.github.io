import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # Target rank
    target_rank = 12
    
    # Precompute indices
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Define the constraints
    # For each i in n, j in m, k in p:
    # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    best_result = None
    best_error = float('inf')
    
    # Try to find a solution for target_rank
    # We will use a simple alternating least squares (ALS) approach.
    # Initialize with random values.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        rank = target_rank
        
        # Initialize
        np.random.seed(attempt)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS iterations
        for iter in range(50):
            if time.time() > deadline:
                break
                
            # Update U
            for u_idx in range(n * m):
                # For each u_idx, we have constraints where u_idx is the U index
                # We need to solve for U[u_idx, :]
                # The terms involving U[u_idx, l] are:
                # sum_l U[u_idx, l] * V[v_idx, l] * W[w_idx, l] = 1
                # This is a linear system for U[u_idx, :]
                # Let x = U[u_idx, :]
                # sum_l x_l * (V[v_idx, l] * W[w_idx, l]) = 1
                # This is a dot product x . (V[v_idx] * W[w_idx]) = 1
                # We have multiple constraints for the same u_idx?
                # Yes, for a fixed u_idx (i,j), there are p values of k.
                # So we have p equations for the rank variables.
                
                # Collect all constraints for this u_idx
                A = []
                b = []
                for (cu, cv, cw) in constraints:
                    if cu == u_idx:
                        vec = V[cv, :] * W[cw, :]
                        A.append(vec)
                        b.append(1.0)
                
                if A:
                    A = np.array(A)
                    b = np.array(b)
                    # Solve A x = b
                    # A is (num_k) x rank
                    # If num_k < rank, underdetermined.
                    # Use least squares
                    x, residuals, rank_u, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[u_idx, :] = x
                    
            # Update V
            for v_idx in range(m * p):
                A = []
                b = []
                for (cu, cv, cw) in constraints:
                    if cv == v_idx:
                        vec = U[cu, :] * W[cw, :]
                        A.append(vec)
                        b.append(1.0)
                if A:
                    A = np.array(A)
                    b = np.array(b)
                    x, residuals, rank_v, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[v_idx, :] = x
                    
            # Update W
            for w_idx in range(p * n):
                A = []
                b = []
                for (cu, cv, cw) in constraints:
                    if cw == w_idx:
                        vec = U[cu, :] * V[cv, :]
                        A.append(vec)
                        b.append(1.0)
                if A:
                    A = np.array(A)
                    b = np.array(b)
                    x, residuals, rank_w, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[w_idx, :] = x
        
        # Check error
        error = 0.0
        for (cu, cv, cw) in constraints:
            val = np.dot(U[cu, :], V[cv, :] * W[cw, :])
            error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # Check if close to half-integers
        if error < 1e-6:
            # Round to nearest 0.5
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            valid = True
            for (cu, cv, cw) in constraints:
                val = np.dot(U_int[cu, :], V_int[cv, :] * W_int[cw, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            
            if valid:
                # Convert to list of lists
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not valid, continue searching
        
    # If no rank-12 solution found, try rank 13? Or just return schoolbook?
    # Given the time, returning schoolbook is safer if 12 fails.
    # But let's try one more thing: maybe the ALS got stuck.
    # I will just return the schoolbook decomposition to guarantee validity.
    
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
    return {"U": U, "V": V, "W": W}
