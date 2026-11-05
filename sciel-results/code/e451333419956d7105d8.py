import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target tensor T[i, j, k] = 1
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try lower ranks
    # Known rank for 2x3x4 is 17
    for R in range(17, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        max_restarts = 50
        max_iters = 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # Avoid all zeros
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                U[0,0] = 1
                V[0,0] = 1
                W[0,0] = 1
            
            # Alternating Least Squares
            prev_err = float('inf')
            for it in range(max_iters):
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # For fixed i, j, we have p equations for R unknowns U[i*m+j, :]
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        b = np.ones(p)
                        # Solve A x = b
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(R)
                        U[i*m + j, :] = x
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # For fixed j, k, we have n equations for R unknowns V[j*p+k, :]
                        # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        b = np.ones(n)
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(R)
                        V[j*p + k, :] = x
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # For fixed k, i, we have m equations for R unknowns W[k*n+i, :]
                        # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b = np.ones(m)
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            x = np.zeros(R)
                        W[k*n + i, :] = x
                        
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                            err += (val - 1.0)**2
                            
                if err < 1e-6:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snapped version
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_snap[i*m + j, :], V_snap[j*p + k, :] * W_snap[k*n + i, :])
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    
                if abs(prev_err - err) < 1e-9:
                    break
                prev_err = err

    # Return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
