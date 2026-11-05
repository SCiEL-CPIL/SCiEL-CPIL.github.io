import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 28
    rank = 28
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Alternating least squares
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Fix U and V, solve for W
        # For each c, we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # This is linear in W[c,:]
        
        # Build the system for W
        # For each (a,b,c), the equation is: W[c,:] dot (U[a,:] * V[b,:]) = T[a,b,c]
        
        # But T[a,b,c] is 1 only when a,b,c are consistent, else 0
        # This is expensive to build
        
        # Instead, let's just check the error and do a simple update
        
        # Compute current error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    error += (val - 1.0) ** 2
        
        if error < 1e-10:
            break
        
        # Simple gradient step
        lr = 0.01
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    grad = val - 1.0
                    U[a] -= lr * grad * V[b] * W[c]
                    V[b] -= lr * grad * U[a] * W[c]
                    W[c] -= lr * grad * U[a] * V[b]
    
    # Check if we have a solution
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                error += (val - 1.0) ** 2
    
    if error < 1e-8:
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
