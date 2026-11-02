import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target Tensor T
    T = np.zeros((size_U, size_V, size_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Known rank bounds: Schoolbook is 30. Lower bound is 15.
    # Literature suggests rank for 2x3x5 is 24.
    # We search from 24 upwards.
    
    possible_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for R in range(24, 31):
        if time.time() > deadline:
            break
            
        # Number of restarts for ALS
        num_restarts = 200 if R < 28 else 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            U = np.random.choice(possible_values, size=(size_U, R))
            V = np.random.choice(possible_values, size=(size_V, R))
            W = np.random.choice(possible_values, size=(size_W, R))
            
            # ALS Iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Solve for W
                # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # For fixed gamma, T[:, :, gamma] is a matrix (size_U, size_V)
                # We want to minimize || T_gamma - (U * V^T) * W_gamma ||_F? 
                # No, T_gamma[alpha, beta] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # This is not a standard linear solve for W_gamma because U and V are coupled in the sum.
                # Wait, the equation is:
                # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # This is a trilinear equation.
                # If we fix U and V, we can solve for W?
                # Let A_gamma be a matrix of size (size_U * size_V) x R.
                # A_gamma[alpha * size_V + beta, r] = U[alpha, r] * V[beta, r]
                # Then A_gamma @ W[gamma, :] = T[:, :, gamma].flatten()
                
                # This is a linear system for each gamma!
                # We can solve for W[gamma, :] using least squares.
                
                W_new = np.zeros((size_W, R))
                for gamma in range(size_W):
                    target = T[:, :, gamma].flatten()
                    A = np.einsum('ar,br->ab', U, V).reshape(-1, R) # This is WRONG.
                    # A[alpha, beta, r] should be U[alpha, r] * V[beta, r]
                    # So A_flat[alpha * size_V + beta, r] = U[alpha, r] * V[beta, r]
                    # np.einsum('ar,br->abr', U, V) gives shape (size_U, size_V, R)
                    A = np.einsum('ar,br->abr', U, V).reshape(-1, R)
                    
                    try:
                        W_gamma, residuals, rank_A, s = np.linalg.lstsq(A, target, rcond=None)
                    except np.linalg.LinAlgError:
                        W_gamma = W[gamma, :]
                    
                    # Snap to half-integers
                    W_gamma_snapped = np.round(W_gamma * 2) / 2.0
                    W_new[gamma, :] = W_gamma_snapped
                
                # Check if W changed significantly? Or just update.
                W = W_new
                
                # Solve for V
                V_new = np.zeros((size_V, R))
                for beta in range(size_V):
                    # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                    # For fixed beta, we have a system for V[beta, r].
                    # Let B_beta be (size_U * size_W) x R.
                    # B_beta[alpha * size_W + gamma, r] = U[alpha, r] * W[gamma, r]
                    # Target is T[:, beta, :].flatten()
                    target = T[:, beta, :].flatten()
                    B = np.einsum('ar,cr->acr', U, W).reshape(-1, R)
                    
                    try:
                        V_beta, residuals, rank_B, s = np.linalg.lstsq(B, target, rcond=None)
                    except np.linalg.LinAlgError:
                        V_beta = V[beta, :]
                    
                    V_beta_snapped = np.round(V_beta * 2) / 2.0
                    V_new[beta, :] = V_beta_snapped
                
                V = V_new
                
                # Solve for U
                U_new = np.zeros((size_U, R))
                for alpha in range(size_U):
                    # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                    # For fixed alpha, system for U[alpha, r].
                    # Let C_alpha be (size_V * size_W) x R.
                    # C_alpha[beta * size_W + gamma, r] = V[beta, r] * W[gamma, r]
                    # Target is T[alpha, :, :].flatten()
                    target = T[alpha, :, :].flatten()
                    C = np.einsum('br,cr->bc', V, W).reshape(-1, R)
                    
                    try:
                        U_alpha, residuals, rank_C, s = np.linalg.lstsq(C, target, rcond=None)
                    except np.linalg.LinAlgError:
                        U_alpha = U[alpha, :]
                    
                    U_alpha_snapped = np.round(U_alpha * 2) / 2.0
                    U_new[alpha, :] = U_alpha_snapped
                
                U = U_new
                
                # Check convergence/exactness
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    converged = True
                    break
            
            if converged:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
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

# EVOLVE-BLOCK-END
