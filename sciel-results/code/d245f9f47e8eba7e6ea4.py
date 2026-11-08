import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 4, 4
    
    # Build the matrix multiplication tensor
    # T[a,b,c] = 1 if a = i*m+k, b = k*p+j, c = j*n+i
    tensor_ones = []
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                tensor_ones.append((a, b, c))
    
    tensor_zeros = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in tensor_ones:
                    tensor_zeros.append((a, b, c))
    
    def check_decomposition(U, V, W, R):
        """Check if the decomposition is valid"""
        errors = []
        for a, b, c in tensor_ones:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
            if abs(val - 1.0) > 1e-6:
                errors.append((a, b, c, val))
        for a, b, c in tensor_zeros:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
            if abs(val) > 1e-6:
                errors.append((a, b, c, val))
        return len(errors) == 0, errors
    
    def objective(x, R):
        """Objective function for optimization"""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R:].reshape(p*n, R)
        
        err = 0.0
        for a, b, c in tensor_ones:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
            err += (val - 1.0) ** 2
        for a, b, c in tensor_zeros:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
            err += val ** 2
        return err
    
    # Try different ranks and find the minimum that works
    for R in range(20, 49):
        # Multiple random restarts
        for trial in range(5):
            np.random.seed(42 + trial)
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            # Use L-BFGS-B for optimization
            res = minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success and res.fun < 1e-8:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                valid, errors = check_decomposition(U, V, W, R)
                if valid:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback: use the trivial rank-48 decomposition
    R = 48
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                r = i * p * m + j * m + k
                U[a, r] = 1
                V[b, r] = 1
                W[c, r] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
