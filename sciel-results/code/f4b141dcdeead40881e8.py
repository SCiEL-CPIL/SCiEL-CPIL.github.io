import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build target tensor indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    num_indices = len(indices)
    
    # Try rank 18 using alternating least squares
    R = 18
    
    # Multiple random initializations
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(seed)
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Solve for W given U, V
            A_W = np.zeros((num_indices, p * n * R))
            for idx, (a, b, c) in enumerate(indices):
                for r in range(R):
                    A_W[idx, c * R + r] = U[a, r] * V[b, r]
            
            try:
                W_flat = np.linalg.lstsq(A_W, np.ones(num_indices), rcond=None)[0]
                W = W_flat.reshape(p * n, R)
            except:
                break
            
            # Solve for U given V, W
            A_U = np.zeros((num_indices, n * m * R))
            for idx, (a, b, c) in enumerate(indices):
                for r in range(R):
                    A_U[idx, a * R + r] = V[b, r] * W[c, r]
            
            try:
                U_flat = np.linalg.lstsq(A_U, np.ones(num_indices), rcond=None)[0]
                U = U_flat.reshape(n * m, R)
            except:
                break
            
            # Solve for V given U, W
            A_V = np.zeros((num_indices, m * p * R))
            for idx, (a, b, c) in enumerate(indices):
                for r in range(R):
                    A_V[idx, b * R + r] = U[a, r] * W[c, r]
            
            try:
                V_flat = np.linalg.lstsq(A_V, np.ones(num_indices), rcond=None)[0]
                V = V_flat.reshape(m * p, R)
            except:
                break
            
            # Check error on non-zero entries
            max_err = 0.0
            for (a, b, c) in indices:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
            
            if max_err < 1e-12:
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify all non-zero entries
                valid = True
                for (a, b, c) in indices:
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
