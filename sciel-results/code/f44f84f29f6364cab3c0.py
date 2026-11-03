# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    
    return max_err < 1e-9

def _compute_error(U, V, W, n, m, p, rank):
    """Compute max error for a decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err

def _adam_local_optimize(U, V, W, n, m, p, rank, max_iters=1000, time_budget=10.0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    mU = np.zeros_like(U_reshaped)
    vU = np.zeros_like(U_reshaped)
    mV = np.zeros_like(V_reshaped)
    vV = np.zeros_like(V_reshaped)
    mW = np.zeros_like(W_reshaped)
    vW = np.zeros_like(W_reshaped)
    
    t = 0
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        t += 1
        
        # Compute Error
        E = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped) - 1.0
        
        # Compute Gradients
        G_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_reshaped, W_reshaped)
        G_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_reshaped, W_reshaped)
        G_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_reshaped, V_reshaped)
        
        # Adam Update
        mU = beta1 * mU + (1 - beta1) * G_U
        vU = beta2 * vU + (1 - beta2) * (G_U ** 2)
        mU_hat = mU / (1 - beta1 ** t)
        vU_hat = vU / (1 - beta2 ** t)
        U_reshaped -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * G_V
        vV = beta2 * vV + (1 - beta2) * (G_V ** 2)
        mV_hat = mV / (1 - beta1 ** t)
        vV_hat = vV / (1 - beta2 ** t)
        V_reshaped -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * G_W
        vW = beta2 * vW + (1 - beta2) * (G_W ** 2)
        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        W_reshaped -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
    U = U_reshaped.reshape(n*m, rank)
    V = V_reshaped.reshape(m*p, rank)
    W = W_reshaped.reshape(p*n, rank)
    
    return U, V, W

def _structured_init_near_schoolbook(n, m, p, target_rank, trial):
    """Initialize near schoolbook but with target_rank columns."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    schoolbook_rank = n * m * p
    
    if target_rank >= schoolbook_rank:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < target_rank:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                col += 1
                
    noise_level = 0.1
    U += np.random.randn(n * m, target_rank) * noise_level
    V += np.random.randn(m * p, target_rank) * noise_level
    W += np.random.randn(p * n, target_rank) * noise_level
    
    return U, V, W

def _try_rank_search(n, m, p, max_rank, time_limit=30.0):
    start_time = time.time()
    deadline = start_time + time_limit
    
    schoolbook_rank = n * m * p
    
    ranks_to_try = list(range(max(1, schoolbook_rank - 10), schoolbook_rank))
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 2.0:
            break
        
        num_tries = 5
        for trial in range(num_tries):
            if time.time() > deadline - 2.0:
                break
            
            U, V, W = _structured_init_near_schoolbook(n, m, p, target_rank, trial)
            
            U, V, W = _adam_local_optimize(U, V, W, n, m, p, target_rank, 
                                           max_iters=1000, time_budget=2.0, lr=0.01)
            
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-9:
                U_list = [[float(x) for x in U[row]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row]] for row in range(p*n)]
                
                U_rounded = [[round(2*x)/2 for x in row] for row in U_list]
                V_rounded = [[round(2*x)/2 for x in row] for row in V_list]
                W_rounded = [[round(2*x)/2 for x in row] for row in W_list]
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None

def _schoolbook_decomposition(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    result = _try_rank_search(n, m, p, schoolbook_rank - 1, time_limit=30.0)
    
    if result is not None:
        return result
    else:
        return schoolbook
# EVOLVE-BLOCK-END
