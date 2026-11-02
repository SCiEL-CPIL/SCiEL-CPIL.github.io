import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (val - 1) ** 2
    return error

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to be safe
    
    best_result = _schoolbook()
    best_rank = best_result["U"][0].__len__()
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # We can try ranks from 100 to best_rank - 1
    for rank in range(100, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Simple local search
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            # Compute error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1) ** 2
            
            if error < 1e-10:
                # Round to integers/half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded
                error_rounded = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                            error_rounded += (val - 1) ** 2
                
                if error_rounded < 1e-10:
                    best_rank = rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break
            
            # Gradient descent step
            # This is a simplified version, actual gradient descent would be complex
            # We just perturb randomly
            U += 0.1 * np.random.randn(n * m, rank)
            V += 0.1 * np.random.randn(m * p, rank)
            W += 0.1 * np.random.randn(p * n, rank)
            
            if error < 1e-10:
                break
    
    return best_result
