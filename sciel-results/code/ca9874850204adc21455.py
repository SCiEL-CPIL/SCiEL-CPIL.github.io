"""Tensor decomposition for 2x3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    # Schoolbook construction: rank = n*m*p = 54
    def schoolbook():
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition is exact
    def verify(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using structured random search
    def try_rank(r, seed_base=0, n_trials=200):
        if time_left() < 5:
            return None
        
        best_err = float('inf')
        best_factors = None
        
        for trial in range(n_trials):
            if time_left() < 5:
                break
            
            np.random.seed(seed_base + trial)
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize columns to help convergence
            for a in range(r):
                nU = np.linalg.norm(U[:, a])
                nV = np.linalg.norm(V[:, a])
                nW = np.linalg.norm(W[:, a])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    U[:, a] /= nU
                    V[:, a] /= nV
                    W[:, a] /= nW
            
            # Alternating optimization: fix two, solve for third
            # This is a non-convex problem, but for small dimensions we can try
            for iteration in range(50):
                if time_left() < 5:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] = argmin sum_k (T[i,j,k] - sum_a U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a])^2
                # This is equivalent to: U[i*m+j, :] = (V^T * W_component)^{-1} * target
                # More precisely, for fixed V, W:
                # T[i,j,:] = U[i*m+j, :] @ (V * W_reshaped)
                # where V * W_reshaped has shape (p, r) with (V*W_reshaped)[k, a] = V[j*p+k, a] * W[k*n+i, a]
                # So U[i*m+j, :] = T[i,j,:] @ pinv(V*W_reshaped)
                
                # Actually, let's use a simpler gradient-free approach:
                # Just compute the error and keep the best
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
                
                # Early stop if we found exact
                if best_err < 1e-10:
                    return [best_factors[0].tolist(), best_factors[1].tolist(), best_factors[2].tolist()]
                
                # Try a simple perturbation-based search
                # This is inefficient but for small r might work
                if iteration % 10 == 5:
                    # Perturb randomly
                    mask = np.random.random(n*m) < 0.1
                    U[mask, :] += np.random.randn(*U[mask, :].shape) * 0.1
                    mask = np.random.random(m*p) < 0.1
                    V[mask, :] += np.random.randn(*V[mask, :].shape) * 0.1
                    mask = np.random.random(p*n) < 0.1
                    W[mask, :] += np.random.randn(*W[mask, :].shape) * 0.1
        
        if best_factors is not None and best_err < 1e-8:
            return [best_factors[0].tolist(), best_factors[1].tolist(), best_factors[2].tolist()]
        
        return None
    
    # Try ranks below schoolbook
    schoolbook_rank = n * m * p
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, schoolbook_rank):
        if time_left() < 10:
            break
        
        result = try_rank(r, seed_base=r * 1000, n_trials=100)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
