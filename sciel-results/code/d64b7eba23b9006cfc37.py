import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Use Kronecker product structure:
    # The 2x3x5 tensor can be decomposed as (2x1x5) ⊗ (1x3x1) ⊗ (1x1x5)
    # Actually, let's think about this differently.
    # 
    # Split the middle dimension (j) into two parts: j=0,1 and j=2
    # C[i,k] = sum_j A[i,j]*B[j,k] = (sum_{j=0,1} A[i,j]*B[j,k]) + A[i,2]*B[2,k]
    #
    # For the first part (j=0,1), we have a 2x2x5 multiplication
    # For the second part (j=2), we have a 2x1x5 multiplication
    #
    # The 2x1x5 tensor has rank 10 (schoolbook: 2*1*5)
    # The 2x2x5 tensor can be decomposed using Strassen for the 2x2x2 part and schoolbook for 2x2x3
    # Actually, let's just use numerical optimization with a better strategy.
    
    def try_rank(target_rank):
        nU = n * m * target_rank
        nV = m * p * target_rank
        nW = p * n * target_rank
        total_vars = nU + nV + nW
        
        def objective_and_gradient(x):
            U = x[:nU].reshape(n*m, target_rank)
            V = x[nU:nU+nV].reshape(m*p, target_rank)
            W = x[nU+nV:].reshape(p*n, target_rank)
            
            S = np.einsum('ar,br,cr->abc', U, V, W)
            error = S - T
            val = np.sum(error**2)
            
            gU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            gV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            gW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return val, grad
        
        import scipy.optimize as opt
        
        # Use multiple random restarts with different strategies
        for r in range(50):
            if time.time() > deadline:
                return None
            
            if r < 10:
                # Small random initialization
                np.random.seed(42 + r)
                x0 = np.random.randn(total_vars) * 0.1
            elif r < 20:
                # Medium initialization
                np.random.seed(100 + r)
                x0 = np.random.randn(total_vars) * 0.5
            else:
                # Large initialization
                np.random.seed(200 + r)
                x0 = np.random.randn(total_vars) * 1.0
            
            try:
                res = opt.minimize(
                    objective_and_gradient,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank 28 first (known to be possible)
    result = try_rank(28)
    if result is not None:
        return result

    # Try rank 29
    result = try_rank(29)
    if result is not None:
        return result

    # Fallback to schoolbook
    return schoolbook()
