import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    for R in range(15, 25):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                    
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_u = np.zeros((p, R))
                        for k in range(p):
                            for c in range(R):
                                A_u[k, c] = W[k*n+i, c] * V[j*p+k, c]
                        b_u = np.ones(p)
                        U_row, residuals, rank, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                        U[i*m+j, :] = U_row
                        
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A_v = np.zeros((n, R))
                        for i in range(n):
                            for c in range(R):
                                A_v[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b_v = np.ones(n)
                        V_row, residuals, rank, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                        V[j*p+k, :] = V_row
                        
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        A_w = np.zeros((m, R))
                        for j in range(m):
                            for c in range(R):
                                A_w[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b_w = np.ones(m)
                        W_row, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                        W[k*n+i, :] = W_row
                        
                # Normalize to prevent drift
                # We can scale columns such that norm(U_col) * norm(V_col) * norm(W_col) = 1?
                # Or just keep them bounded.
                
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(R):
                            val += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                        error += (val - 1)**2
                        
            if error < 1e-6:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(R):
                                val += W_rounded[k*n+i, c] * U_rounded[i*m+j, c] * V_rounded[j*p+k, c]
                            if abs(val - 1) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                if exact:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
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
    return {"U": U, "V": V, "W": W}
