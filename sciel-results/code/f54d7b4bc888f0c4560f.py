"""Combined program for Matrix Multiplication 3,4,4 rank minimization.
Implements schoolbook decomposition and a heuristic search for lower rank.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Construct Schoolbook Solution (Rank 48)
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1.0
                V_sb[j * p + k, idx] = 1.0
                W_sb[k * n + i, idx] = 1.0
                idx += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Vectorized Verification
    def verify(U, V, W, tol=1e-9):
        # U: (n*m, R) -> (n, m, R)
        # V: (m*p, R) -> (m, p, R)
        # W: (p*n, R) -> (p, n, R)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # val[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        val = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.all(np.abs(val - 1.0) <= tol)

    # Vectorized Loss and Gradient
    def compute_loss_and_grad(U, V, W):
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        val = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        err = val - 1.0
        L = np.sum(err * err)
        
        # Gradients
        # dL/dU[i,j,c] = 2 * sum_k err[i,j,k] V[j,k,c] W[k,i,c]
        gU_t = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, V_t, W_t)
        # dL/dV[j,k,c] = 2 * sum_i err[i,j,k] U[i,j,c] W[k,i,c]
        gV_t = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, U_t, W_t)
        # dL/dW[k,i,c] = 2 * sum_j err[i,j,k] U[i,j,c] V[j,k,c]
        gW_t = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, U_t, V_t)
        
        return L, gU_t.reshape(n*m, -1), gV_t.reshape(m*p, -1), gW_t.reshape(p*n, -1)

    def adam_step(params, grads, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        for i, param in enumerate(params):
            if i not in state:
                state[i] = {'m': np.zeros_like(param), 'v': np.zeros_like(param), 't': 0}
            st = state[i]
            st['t'] += 1
            st['m'] = beta1 * st['m'] + (1 - beta1) * grads[i]
            st['v'] = beta2 * st['v'] + (1 - beta2) * (grads[i] ** 2)
            m_hat = st['m'] / (1 - beta1 ** st['t'])
            v_hat = st['v'] / (1 - beta2 ** st['t'])
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def try_rank(R, num_trials=5, max_steps=500):
        if R <= 0:
            return None
        if time_left() < 5:
            return None
            
        for trial in range(num_trials):
            if time_left() < 2:
                break
                
            # Random initialization
            # Use a mix of small random and some structured init?
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale initialization
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            params = [U, V, W]
            state = {}
            lr = 0.05
            prev_loss = float('inf')
            
            for step in range(max_steps):
                if time_left() < 2:
                    break
                    
                L, gU, gV, gW = compute_loss_and_grad(U, V, W)
                grads = [gU, gV, gW]
                
                # Check convergence
                if L < 1e-8:
                    break
                if prev_loss - L < 1e-10 and L < 1e-4:
                    break
                prev_loss = L
                
                # Adam step
                adam_step(params, grads, state, lr=lr)
                
            # Verify and snap
            if verify(U, V, W):
                # Round to nearest integer/half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
            else:
                # If continuous verify fails, try snapping anyway?
                # Sometimes float error accumulates.
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
            
        return None

    # Try to find a lower rank
    # Focus on rank 47 and 46 primarily.
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 0), -1):
        if time_left() < 10:
            break
            
        trials = 20
        steps = 200
        
        result = try_rank(R, num_trials=trials, max_steps=steps)
        if result:
            best_U, best_V, best_W = result
            best_rank = R
            if time_left() < 30:
                break

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
