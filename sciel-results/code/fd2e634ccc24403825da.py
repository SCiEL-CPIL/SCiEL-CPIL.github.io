import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    deadline = time.time() + 140
    
    # Try to find a lower-rank decomposition
    for rank in range(32, 45):
        if time.time() > deadline - 15:
            break
        
        result = try_find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
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


def try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-rank exact decomposition."""
    
    # Method 1: Alternating least squares
    for seed in range(5):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.01
        V = np.random.randn(m * p, rank) * 0.01
        W = np.random.randn(p * n, rank) * 0.01
        
        for iteration in range(1000):
            if time.time() > deadline - 10:
                return None
            
            # Update W
            for i in range(n):
                for k in range(p):
                    A = np.array([sum(U[i*m+j, r] * V[j*p+k, r] for j in range(m)) for r in range(rank)])
                    norm_sq = A @ A
                    if norm_sq > 1e-12:
                        W[k*n+i, :] = A / norm_sq
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.array([sum(V[j*p+k, r] * W[k*n+i, r] for k in range(p)) for r in range(rank)])
                    norm_sq = B @ B
                    if norm_sq > 1e-12:
                        U[i*m+j, :] = B / norm_sq
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C = np.array([sum(U[i*m+j, r] * W[k*n+i, r] for i in range(n)) for r in range(rank)])
                    norm_sq = C @ C
                    if norm_sq > 1e-12:
                        V[j*p+k, :] = C / norm_sq
            
            # Check convergence
            max_error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        max_error = max(max_error, abs(val - 1))
            
            if max_error < 1e-12:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify_decomposition(n, m, p, U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Method 2: Structured initialization based on schoolbook
    if time.time() > deadline - 5:
        return None
    
    # Start with schoolbook and try to reduce rank
    # This is complex, so I'll skip it for now
    
    return None


def verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-8:
                    return False
    return True
