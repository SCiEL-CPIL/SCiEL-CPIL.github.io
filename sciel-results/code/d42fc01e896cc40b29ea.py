import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Build target tensor T of shape (N_U, N_V, N_W)
    T = np.zeros((N_U, N_V, N_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try ranks from lower bound upwards
    # Lower bound is max(n*m, m*p, n*p) = 12
    # Known rank for 2x3x4 is likely 19
    # We will try 12 to 24
    best_rank = 24
    best_decomp = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for R in range(12, 25):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple random restarts for each rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(N_U, R)
            V = np.random.randn(N_V, R)
            W = np.random.randn(N_W, R)
            
            # ALS Iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # A shape: (N_U * N_V * N_W, N_U * R)
                # b shape: (N_U * N_V * N_W)
                A_U = np.zeros((N_U * N_V * N_W, N_U * R))
                b_U = T.flatten()
                
                for a in range(N_U):
                    for b in range(N_V):
                        for c in range(N_W):
                            idx = a * N_V * N_W + b * N_W + c
                            coeff = V[b, :] * W[c, :] # shape (R,)
                            # U[a, r] corresponds to column a*R + r
                            A_U[idx, a*R : (a+1)*R] = coeff
                            
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(N_U, R)
                
                # Solve for V
                A_V = np.zeros((N_U * N_V * N_W, N_V * R))
                for a in range(N_U):
                    for b in range(N_V):
                        for c in range(N_W):
                            idx = a * N_V * N_W + b * N_W + c
                            coeff = U[a, :] * W[c, :]
                            A_V[idx, b*R : (b+1)*R] = coeff
                            
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_U, rcond=None)
                V = V_flat.reshape(N_V, R)
                
                # Solve for W
                A_W = np.zeros((N_U * N_V * N_W, N_W * R))
                for a in range(N_U):
                    for b in range(N_V):
                        for c in range(N_W):
                            idx = a * N_V * N_W + b * N_W + c
                            coeff = U[a, :] * V[b, :]
                            A_W[idx, c*R : (c+1)*R] = coeff
                            
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_U, rcond=None)
                W = W_flat.reshape(N_W, R)
                
            # Check solution
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            error = 0.0
            for a in range(N_U):
                for b in range(N_V):
                    for c in range(N_W):
                        val = np.sum(U_r[a, :] * V_r[b, :] * W_r[c, :])
                        target = T[a, b, c]
                        if abs(val - target) > 1e-6:
                            error += abs(val - target)
                            
            if error < 1e-6:
                found = True
                best_rank = R
                best_decomp = (U_r, V_r, W_r)
                break # Break restarts
                
        if found:
            break # Break rank loop
            
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to lists of lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
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
        return {"U": U, "V": V, "W": W}
