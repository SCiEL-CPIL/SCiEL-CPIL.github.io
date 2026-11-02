import numpy as np
import time
import sys

# DIMENSIONS
n, m, p = 3, 3, 5

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    current_time = time.time()
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Precompute the target tensor T as a function or sparse dict
    # T[I, J, K] = 1 if I=i*m+j, J=j'*p+k, K=k'*n+i' and j==j', i==i', k==k'
    # We can generate the non-zero entries
    # I in [0, n*m), J in [0, m*p), K in [0, p*n)
    # I = i*m + j
    # J = j'*p + k
    # K = k'*n + i'
    # Condition: j == j', i == i', k == k'
    
    # Let's create a list of (I, J, K) indices where T is 1.
    # And a list of all indices to compute residual?
    # Total elements in tensor: (n*m) * (m*p) * (p*n) = 9 * 15 * 15 = 2025.
    # This is small. We can store T as a dense array for speed.
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Function to compute residual given U, V, W
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Residual R[I,J,K] = sum_c U[I,c] V[J,c] W[K,c] - T[I,J,K]
    # We want sum of squares of R.
    
    def compute_residual(U, V, W):
        # U @ (V * W.T)? No.
        # R = np.einsum('ic,jc,kc->ijk', U, V, W)
        # This might be slow in pure Python/Numpy if r is large, but r <= 45.
        # 9*15*15*45 operations = ~90k flops per residual calc. Very fast.
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R

    def objective_and_grad(params, r):
        # params is flat array of size r*(n*m + m*p + p*n)
        # Order: U (n*m*r), V (m*p*r), W (p*n*r)
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n*m, r)
        V = params[size_U:size_U+size_V].reshape(m*p, r)
        W = params[size_U+size_V:].reshape(p*n, r)
        
        R = compute_residual(U, V, W)
        loss = np.sum(R**2)
        
        # Gradient
        # dL/dU[I,c] = 2 * sum_{J,K} R[I,J,K] V[J,c] W[K,c]
        # dL/dV[J,c] = 2 * sum_{I,K} R[I,J,K] U[I,c] W[K,c]
        # dL/dW[K,c] = 2 * sum_{I,J} R[I,J,K] U[I,c] V[J,c]
        
        # Using einsum for gradient
        # grad_U[I,c] = 2 * einsum('ijk,jc,kc->ic', R, V, W)
        # grad_V[J,c] = 2 * einsum('ijk,ic,kc->jc', R, U, W)
        # grad_W[K,c] = 2 * einsum('ijk,ic,jc->kc', R, U, V)
        
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    def check_exactness(U, V, W):
        # Check if entries are close to half-integers
        # And if the residual with rounded entries is 0
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounding changed the values significantly
        # If not, we are close to a half-integer solution
        if np.max(np.abs(U - U_rounded)) > 1e-5:
            return False, None, None, None
        if np.max(np.abs(V - V_rounded)) > 1e-5:
            return False, None, None, None
        if np.max(np.abs(W - W_rounded)) > 1e-5:
            return False, None, None, None
            
        R_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(R_rounded)) < 1e-9:
            return True, U_rounded, V_rounded, W_rounded
        return False, None, None, None

    # Search for minimal rank
    # Start from a reasonable lower bound.
    # For 3x3x3, rank is 23.
    # For 3x3x5, let's guess around 30-35.
    # We will try ranks from 25 to 45.
    
    best_result = None
    
    # To save time, we can start from a higher rank if we suspect low rank is hard to find,
    # but starting low guarantees minimality if found.
    # However, optimization might get stuck in local minima for low ranks.
    # We will use multiple restarts for each rank.
    
    for r in range(25, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of restarts depends on remaining time
        # Let's do 5 restarts per rank initially, more if time allows.
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Small random values to avoid symmetry breaking issues?
            # Or random normal.
            np.random.seed(42 + restart + r)
            params = np.random.randn(r * (n*m + m*p + p*n)) * 0.1
            
            # Use scipy minimize
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r)[1],
                    method='BFGS',
                    options={'gtol': 1e-8, 'maxiter': 200}
                )
                # Check success
                if res.success or res.fun < 1e-8:
                    U, V, W = res.x[:n*m*r].reshape(n*m, r), \
                              res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r), \
                              res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    is_exact, U_r, V_r, W_r = check_exactness(U, V, W)
                    if is_exact:
                        # Convert to lists of integers/half-integers
                        # The prompt asks for integer/half-integer entries.
                        # We can return them as floats or ints?
                        # "integer/half-integer entries"
                        # We'll return floats that are exactly k/2.
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
            except Exception as e:
                pass

    # If no low rank found, return schoolbook
    rank = rank_schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
