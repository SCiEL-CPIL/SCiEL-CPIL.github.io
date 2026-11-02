"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if valid else 0."""
    return np.ones((n, m, p))

def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct tensor from factor matrices. T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T

def _validate(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _snap_to_half_integers(U, V, W, tol=0.01):
    """Snap values close to half-integers. Returns None if snap is not close enough."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    if (np.max(np.abs(U - U_s)) < tol and
        np.max(np.abs(V - V_s)) < tol and
        np.max(np.abs(W - W_s)) < tol):
        U_s = np.round(U_s * 2) / 2.0
        V_s = np.round(V_s * 2) / 2.0
        W_s = np.round(W_s * 2) / 2.0
        return U_s, V_s, W_s
    return None

def _try_random_decomposition(n, m, p, max_rank, seed, time_limit):
    """Try to find a decomposition of rank max_rank using random init + gradient descent."""
    rng = np.random.RandomState(seed)
    T_target = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    start_time = time.time()
    
    for attempt in range(20):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Random initialization
        U = rng.randn(n*m, max_rank) * 0.5
        V = rng.randn(m*p, max_rank) * 0.5
        W = rng.randn(p*n, max_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for step in range(800):
            if time.time() - start_time > time_limit - 5:
                break
            
            T = _reconstruct(U, V, W, n, m, p, max_rank)
            err = np.sum((T - T_target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients using einsum for speed
            dT = 2 * (T - T_target)
            dT_r = dT.reshape(n, m, p)
            
            V_r = V.reshape(m, p, max_rank)
            W_r = W.reshape(p, n, max_rank)
            U_r = U.reshape(n, m, max_rank)
            
            dU_r = np.einsum('ijk,jkc,kic->ijc', dT_r, V_r, W_r)
            dV_r = np.einsum('ijk,ijc,kic->jkc', dT_r, U_r, W_r)
            dW_r = np.einsum('ijk,ijc,jkc->kic', dT_r, U_r, V_r)
            
            dU = dU_r.reshape(n*m, max_rank)
            dV = dV_r.reshape(m*p, max_rank)
            dW = dW_r.reshape(p*n, max_rank)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if step % 50 == 0:
                lr *= 0.95
        
        if best_err < 1e-12:
            break
    
    if best_U is not None and best_err < 1e-9:
        snapped = _snap_to_half_integers(best_U, best_V, best_W)
        if snapped is not None:
            U_s, V_s, W_s = snapped
            if _validate(U_s, V_s, W_s, n, m, p, max_rank):
                return U_s, V_s, W_s
    
    return None

def _try_lsa_decomposition(n, m, p, max_rank, seed, time_limit):
    """Try to find a decomposition using a more structured approach."""
    rng = np.random.RandomState(seed)
    T_target = _build_tensor(n, m, p)
    
    start_time = time.time()
    
    # Try multiple random initializations with different strategies
    for attempt in range(15):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Strategy 1: Random init
        U = rng.randn(n*m, max_rank) * 0.3
        V = rng.randn(m*p, max_rank) * 0.3
        W = rng.randn(p*n, max_rank) * 0.3
        
        # Adam-like optimizer
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = 0.02
        
        for step in range(1000):
            if time.time() - start_time > time_limit - 5:
                break
            
            T = _reconstruct(U, V, W, n, m, p, max_rank)
            err = np.sum((T - T_target)**2)
            
            if err < 1e-12:
                break
            
            dT = 2 * (T - T_target)
            dT_r = dT.reshape(n, m, p)
            
            V_r = V.reshape(m, p, max_rank)
            W_r = W.reshape(p, n, max_rank)
            U_r = U.reshape(n, m, max_rank)
            
            dU_r = np.einsum('ijk,jkc,kic->ijc', dT_r, V_r, W_r)
            dV_r = np.einsum('ijk,ijc,kic->jkc', dT_r, U_r, W_r)
            dW_r = np.einsum('ijk,ijc,jkc->kic', dT_r, U_r, V_r)
            
            dU = dU_r.reshape(n*m, max_rank)
            dV = dV_r.reshape(m*p, max_rank)
            dW = dW_r.reshape(p*n, max_rank)
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * dU**2
            m_U_hat = m_U / (1 - beta1**(step + 1))
            v_U_hat = v_U / (1 - beta2**(step + 1))
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV**2
            m_V_hat = m_V / (1 - beta1**(step + 1))
            v_V_hat = v_V / (1 - beta2**(step + 1))
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW**2
            m_W_hat = m_W / (1 - beta1**(step + 1))
            v_W_hat = v_W / (1 - beta2**(step + 1))
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        # Check result
        T_final = _reconstruct(U, V, W, n, m, p, max_rank)
        err_final = np.sum((T_final - T_target)**2)
        if err_final < 1e-9:
            snapped = _snap_to_half_integers(U, V, W)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                if _validate(U_s, V_s, W_s, n, m, p, max_rank):
                    return U_s, V_s, W_s
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 145  # Leave margin before 150s kill
    
    # Try ranks from lower to higher
    for rank in range(schoolbook_rank - 1, max(15, schoolbook_rank - 8), -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        rank_time_limit = min(18, time_limit - (time.time() - start_time))
        if rank_time_limit <= 8:
            break
        
        # Try random decomposition first
        result = _try_random_decomposition(n, m, p, rank, seed=rank * 1000 + 42, time_limit=rank_time_limit // 2)
        if result is not None:
            U_s, V_s, W_s = result
            U_list = [[val for val in row] for row in U_s.tolist()]
            V_list = [[val for val in row] for row in V_s.tolist()]
            W_list = [[val for val in row] for row in W_s.tolist()]
            if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
        
        # Try LSA-style decomposition if time remains
        remaining = rank_time_limit - (time.time() - start_time)
        if remaining > 5:
            result = _try_lsa_decomposition(n, m, p, rank, seed=rank * 2000 + 17, time_limit=remaining)
            if result is not None:
                U_s, V_s, W_s = result
                U_list = [[val for val in row] for row in U_s.tolist()]
                V_list = [[val for val in row] for row in V_s.tolist()]
                W_list = [[val for val in row] for row in W_s.tolist()]
                if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
