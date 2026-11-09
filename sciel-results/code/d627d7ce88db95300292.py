import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 5
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Indices for the equations
    # i in 0..2, j in 0..2, k in 0..4
    idx_i = []
    idx_j = []
    idx_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_i.append(i * m + j)
                idx_j.append(j * p + k)
                idx_k.append(k * n + i)
                
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    def error_func(params, r):
        # params is a flat vector of size 39*r
        U = params[:size_U*r].reshape(size_U, r)
        V = params[size_U*r:size_U*r+size_V*r].reshape(size_V, r)
        W = params[size_U*r+size_V*r:].reshape(size_W, r)
        
        # Compute the sum for each equation
        # U[idx_i, c] * V[idx_j, c] * W[idx_k, c]
        # We need sum over c
        # This can be done efficiently with matrix multiplication or einsum
        # But r is small, so loop is fine
        sums = np.zeros(len(idx_i))
        for c in range(r):
            sums += U[idx_i, c] * V[idx_j, c] * W[idx_k, c]
            
        errors = sums - 1.0
        return np.sum(errors**2)
        
    def gradient_func(params, r):
        # Analytical gradient
        U = params[:size_U*r].reshape(size_U, r)
        V = params[size_U*r:size_U*r+size_V*r].reshape(size_V, r)
        W = params[size_U*r+size_V*r:].reshape(size_W, r)
        
        sums = np.zeros(len(idx_i))
        for c in range(r):
            sums += U[idx_i, c] * V[idx_j, c] * W[idx_k, c]
            
        errors = sums - 1.0
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for c in range(r):
            # d/dU[idx_i, c] sum errors^2
            # 2 * errors * V[idx_j, c] * W[idx_k, c]
            term = 2 * errors * V[idx_j, c] * W[idx_k, c]
            # Scatter add to gU
            np.add.at(gU, (idx_i, c), term)
            
            term = 2 * errors * U[idx_i, c] * W[idx_k, c]
            np.add.at(gV, (idx_j, c), term)
            
            term = 2 * errors * U[idx_i, c] * V[idx_j, c]
            np.add.at(gW, (idx_k, c), term)
            
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
    def check_solution(U, V, W):
        # Check if entries are close to half-integers
        # And if the equations are satisfied exactly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check half-integer
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_rank = 45
    best_solution = None
    
    # Try to find a solution with rank < 45
    for r in range(44, 19, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use a mix of random and structured initialization
            # Structured: start with naive and zero out random columns
            if restart == 0:
                # Naive initialization
                U = np.zeros((size_U, r))
                V = np.zeros((size_V, r))
                W = np.zeros((size_W, r))
                # Fill first 45 columns if r >= 45, but r < 45 here
                # So we can't use naive directly.
                # Instead, use random
                U = np.random.randn(size_U, r)
                V = np.random.randn(size_V, r)
                W = np.random.randn(size_W, r)
            else:
                U = np.random.randn(size_U, r) * 0.1
                V = np.random.randn(size_V, r) * 0.1
                W = np.random.randn(size_W, r) * 0.1
                
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                res = minimize(error_func, params, args=(r,), jac=gradient_func, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.success:
                U_opt = res.x[:size_U*r].reshape(size_U, r)
                V_opt = res.x[size_U*r:size_U*r+size_V*r].reshape(size_V, r)
                W_opt = res.x[size_U*r+size_V*r:].reshape(size_W, r)
                
                # Round
                U_round = round_to_half(U_opt)
                V_round = round_to_half(V_opt)
                W_round = round_to_half(W_opt)
                
                if check_solution(U_round, V_round, W_round):
                    best_rank = r
                    best_solution = (U_round, V_round, W_round)
                    found = True
                    break # Found a solution for this rank, go to next lower rank
        
        if found:
            continue # Try lower rank
            
    # If no solution found, return naive
    if best_solution is None:
        rank = 45
        U = np.zeros((size_U, rank))
        V = np.zeros((size_V, rank))
        W = np.zeros((size_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
