import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Target tensor T[i, j, k, l] corresponds to A[i, j] * B[k, l] contributing to C[i, l] if j == k
    # We want sum_r U[i, j, r] * V[k, l, r] * W[i, l, r] = delta[j, k] * (A[i,j]B[k,l] coeff)
    # Actually, the tensor M has indices (i, j) for A, (k, l) for B, and output (i, l).
    # M_{i, j, k, l} = 1 if j == k, else 0? No.
    # C_{i, l} = sum_j A_{i, j} B_{j, l}.
    # So the term A_{i, j} B_{k, l} appears in C_{i, l} with coefficient 1 if j == k, else 0.
    # So Target T[i, j, k, l] = 1 if j == k else 0.
    # We decompose T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r is a vector of size n*m. V_r is size m*p. W_r is size p*n.
    # T[i, j, k, l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # Note: W index is k*n+i in the prompt? 
    # Prompt: W[k*n+i, r]. 
    # My W index: output is C[i, l]. So W should depend on i and l.
    # Prompt says W shape (p*n, rank). Index is k*n+i? 
    # Wait, the prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # If W index is k*n+i, then k is from 0..p-1, i from 0..n-1.
    # But C is n x p. Indices are i (0..n-1) and k (0..p-1).
    # So W[k*n+i, r] corresponds to C[i, k].
    # So the decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k] = sum_r W[k*n+i, r] * (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k])?
    # No, the linear forms are over all indices.
    # x_r = sum_{i, j} U[i*m+j, r] A[i, j]
    # y_r = sum_{j, k} V[j*p+k, r] B[j, k]
    # C[i, k] = sum_r W[k*n+i, r] x_r y_r
    # This matches the standard tensor decomposition.
    
    def target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # A[i, j] and B[j, k] contribute to C[i, k]
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    T = target_tensor()
    
    # Try to find rank 20 decomposition
    rank = 20
    best_result = None
    best_score = 1.0 # Schoolbook is 27, so 27/27 = 1.0. If we find 20, score is 27/20 = 1.35.
    
    # If we fail, we return schoolbook
    def get_schoolbook():
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Optimization
    # We want min || sum_r u_r v_r^T w_r - T ||_F
    # This is a non-convex problem.
    # We can use random restarts and gradient descent.
    
    # To make it faster, we can fix W to random values?
    # Or use a smarter initialization.
    
    # Given the time limit, I will try a few random restarts with scipy.optimize.
    
    import scipy.optimize as so
    
    def error(params):
        # params: flattened U, V, W
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        # Compute decomposition
        # T_est[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Using einsum: 'ir, jr, kr -> ijk'
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        diff = T_est - T
        return np.sum(diff**2)

    # Gradient is needed for speed.
    def gradient(params):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T_est - T
        
        # dL/dU = 2 * diff * V * W
        # diff is (i, j, k). V is (j, r). W is (k, r).
        # grad_U[i, r] = sum_{j, k} diff[i, j, k] V[j, r] W[k, r]
        grad_U = np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        grad_U *= 2
        grad_V *= 2
        grad_W *= 2
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Let's try values in [-1, 1]
        np.random.seed(restart)
        U_init = np.random.randn(n*m, rank)
        V_init = np.random.randn(m*p, rank)
        W_init = np.random.randn(p*n, rank)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B
        try:
            res = so.minimize(error, params_init, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success:
                # Check error
                err = res.fun
                if err < 1e-6:
                    # Extract matrices
                    size_u = n * m * rank
                    size_v = m * p * rank
                    size_w = p * n * rank
                    
                    U_opt = res.x[:size_u].reshape((n*m, rank))
                    V_opt = res.x[size_u:size_u+size_v].reshape((m*p, rank))
                    W_opt = res.x[size_u+size_v:].reshape((p*n, rank))
                    
                    # Round to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    T_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.allclose(T_check, T, atol=1e-9):
                        # Found valid rank 20 decomposition
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
        except Exception as e:
            continue

    # Fallback to schoolbook
    return get_schoolbook()
