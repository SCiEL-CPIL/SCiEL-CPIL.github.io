import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    I, J, K = T.shape
    
    def verify_decomposition(U, V, W):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_hat - T)
        if np.max(diff) < 1e-9:
            return True
        return False

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def loss_and_grad(params, rank):
        size_U = I * rank
        size_V = J * rank
        size_W = K * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(I, rank)
        V = V_flat.reshape(J, rank)
        W = W_flat.reshape(K, rank)
        
        # Compute T_hat
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual E = T - T_hat
        E = T - T_hat
        loss = np.sum(E**2)
        
        # Regularization
        reg_U = np.sum(np.sin(np.pi * 2 * U)**2)
        reg_V = np.sum(np.sin(np.pi * 2 * V)**2)
        reg_W = np.sum(np.sin(np.pi * 2 * W)**2)
        reg = reg_U + reg_V + reg_W
        
        total_loss = loss + 0.01 * reg
        
        # Gradient
        # dLoss/dU = -2 * E * (V W) + 0.01 * dReg/dU
        # E is (I, J, K)
        # We need sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        # This is a tensor contraction.
        # grad_U[a,r] = -2 * sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        
        # Efficiently compute grad_U
        # E * V: (I, J, K) * (J, R) -> (I, K, R) ?
        # Let's use einsum.
        # grad_U = -2 * np.einsum('abc,br,cr->ar', E, V, W)
        # grad_V = -2 * np.einsum('abc,ar,cr->br', E, U, W)
        # grad_W = -2 * np.einsum('abc,ar,br->cr', E, U, V)
        
        grad_U = -2 * np.einsum('abc,br,cr->ar', E, V, W)
        grad_V = -2 * np.einsum('abc,ar,cr->br', E, U, W)
        grad_W = -2 * np.einsum('abc,ar,br->cr', E, U, V)
        
        # Regularization gradients
        # d/dx sin^2(2 pi x) = 2 sin(2 pi x) cos(2 pi x) * 2 pi = 2 pi sin(4 pi x)
        # So dReg/dU = 0.01 * 2 pi sin(4 pi U)
        grad_reg_U = 0.01 * 2 * np.pi * np.sin(4 * np.pi * U)
        grad_reg_V = 0.01 * 2 * np.pi * np.sin(4 * np.pi * V)
        grad_reg_W = 0.01 * 2 * np.pi * np.sin(4 * np.pi * W)
        
        grad_U += grad_reg_U
        grad_V += grad_reg_V
        grad_W += grad_reg_W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return total_loss, grad

    def solve_rank(rank, max_restarts=100):
        np.random.seed(42 + rank)
        best_loss = float('inf')
        best_params = None
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly
            U0 = np.random.randn(I, rank) * 0.5
            V0 = np.random.randn(J, rank) * 0.5
            W0 = np.random.randn(K, rank) * 0.5
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(loss_and_grad, params0, args=(rank,), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            if time.time() > deadline - 10:
                break
                
        if best_params is None:
            return None
            
        size_U = I * rank
        size_V = J * rank
        U = best_params[:size_U].reshape(I, rank)
        V = best_params[size_U:size_U+size_V].reshape(J, rank)
        W = best_params[size_U+size_V:].reshape(K, rank)
        
        U_rounded = round_to_half_int(U)
        V_rounded = round_to_half_int(V)
        W_rounded = round_to_half_int(W)
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
        return None

    result = solve_rank(29)
    if result:
        return result
        
    if time.time() < deadline - 30:
        result = solve_rank(28, max_restarts=50)
        if result:
            return result

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
