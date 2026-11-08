import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    # Known bounds for 3x3x3 are ~23. For 3x3x4, it might be lower or similar.
    # Let's try a range.
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try ranks from a lower bound upwards
    # Lower bound for rank of M_{n,m,p} is max(n,m,p) = 4.
    # But practically, it's higher.
    # Let's try 18 to 36.
    
    best_rank = n * m * p
    best_result = None
    # Initialize with schoolbook to have a valid fallback
    U_init = np.zeros((n*m, best_rank))
    V_init = np.zeros((m*p, best_rank))
    W_init = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, col] = 1
                V_init[j*p+k, col] = 1
                W_init[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
    
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small integers to encourage integer solutions
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            # Alternating minimization
            converged = False
            for it in range(50):
                # Update W
                # For each k, i
                for k in range(p):
                    for i in range(n):
                        # Build A (3 x r) and b (3)
                        A = np.zeros((m, r))
                        for j in range(m):
                            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        # Solve A w = b for minimum norm w
                        # w = A^T (A A^T)^-1 b
                        try:
                            AA_T = A @ A.T
                            # Add small regularization for stability
                            AA_T += 1e-6 * np.eye(m)
                            w = A.T @ np.linalg.solve(AA_T, b)
                            W[k*n+i, :] = w
                        except np.linalg.LinAlgError:
                            # Fallback to pseudo-inverse
                            w = np.linalg.pinv(A) @ b
                            W[k*n+i, :] = w
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            AA_T = A @ A.T
                            AA_T += 1e-6 * np.eye(p)
                            u = A.T @ np.linalg.solve(AA_T, b)
                            U[i*m+j, :] = u
                        except np.linalg.LinAlgError:
                            u = np.linalg.pinv(A) @ b
                            U[i*m+j, :] = u
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            AA_T = A @ A.T
                            AA_T += 1e-6 * np.eye(n)
                            v = A.T @ np.linalg.solve(AA_T, b)
                            V[j*p+k, :] = v
                        except np.linalg.LinAlgError:
                            v = np.linalg.pinv(A) @ b
                            V[j*p+k, :] = v
                
                # Check residual
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res += (val - 1.0)**2
                
                if res < 1e-4:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exact decomposition with rounded values
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        best_rank = r
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        found = True
                        break
                if time.time() > deadline:
                    break
            
            if found:
                break
    
    return best_result
