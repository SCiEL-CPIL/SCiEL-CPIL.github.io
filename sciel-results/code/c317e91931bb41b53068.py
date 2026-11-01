import numpy as np
import time
import sys

_START = time.time()
_DEADLINE = _START + 140.0

def _check_deadline():
    if time.time() > _DEADLINE - 5.0:
        return True
    return False

def _build_tensor_2x3x4():
    n, m, p = 2, 3, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1.0
    return T

def _try_find_rank_12_2x3x4():
    """
    Attempts to find a rank-12 decomposition for 2x3 * 3x4.
    Returns U, V, W if found, else None.
    """
    n, m, p = 2, 3, 4
    rank = 12
    T = _build_tensor_2x3x4()
    
    # Initialize U, V, W randomly
    # Scale matters for gradient descent
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Use a simple gradient descent step
    # Loss: || T - sum(U_t V_t W_t) ||^2
    # Derivative wrt U: sum( (T_decomp - T)_{b,d} V_{b,t} W_{d,t} )
    
    steps = 2000
    lr = 0.05
    
    for step in range(steps):
        if _check_deadline():
            return None
            
        T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
        err = T_decomp - T
        
        # Gradients
        # dU[a,t] = sum_{b,d} err[a,b,d] V[b,t] W[d,t]
        dU = np.einsum('abd,bt,dt->at', err, V, W, optimize=True)
        dV = np.einsum('abd,at,dt->bt', err, U, W, optimize=True)
        dW = np.einsum('abd,at,bt->dt', err, U, V, optimize=True)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Periodic check
        if step % 100 == 0:
            loss = np.sum(err**2)
            if loss < 1e-10:
                # Try to round
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_check = np.einsum('at,bt,dt->abd', U_int, V_int, W_int, optimize=True)
                if np.allclose(T_check, T, atol=1e-10, rtol=0):
                    return U_int, V_int, W_int
                    
    # Final check
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    if np.allclose(T_decomp, T, atol=1e-8, rtol=0):
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        T_check = np.einsum('at,bt,dt->abd', U_int, V_int, W_int, optimize=True)
        if np.allclose(T_check, T, atol=1e-10, rtol=0):
            return U_int, V_int, W_int
            
    return None

def _build_sub_2x3_3x2():
    """Rank-7 decomposition for 2x3 * 3x2 = 2x2."""
    U = np.zeros((6, 7))
    V = np.zeros((6, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = -1; W[1, 0] = -1; W[3, 0] = 1
    
    U[1, 1] = 1
    V[2, 1] = 1
    W[0, 1] = 1
    
    U[2, 2] = 1
    V[4, 2] = 1
    W[0, 2] = 1
    
    U[5, 3] = 1
    V[5, 3] = 1
    W[3, 3] = 1
    
    U[0, 4] = 1; U[4, 4] = -1
    V[0, 4] = 1; V[3, 4] = -1
    W[0, 4] = 1; W[2, 4] = 1; W[1, 4] = 1; W[3, 4] = 1
    
    U[3, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    U[1, 6] = 1; U[5, 6] = -1
    V[2, 6] = 1; V[5, 6] = -1
    W[3, 6] = -1
    
    return U, V, W

def _fallback_28():
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _build_sub_2x3_3x2()
    r_sub = 7
    num_blocks = 4
    total_rank = num_blocks * r_sub
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        for j in range(m):
            for k_prime in range(2):
                b_global = j * p + (t * 2 + k_prime)
                b_local = j * 2 + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        for k_prime in range(2):
            for i in range(n):
                d_global = (t * 2 + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
                
    return U_full, V_full, W_full

def construct():
    n, m, p = 2, 3, 8
    
    # Try to find rank 12 for 2x3x4
    if not _check_deadline():
        res = _try_find_rank_12_2x3x4()
        if res is not None:
            U_sub, V_sub, W_sub = res
            r_sub = 12
            num_blocks = 2 # 8 / 4
            total_rank = num_blocks * r_sub # 24
            
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            for t in range(num_blocks):
                U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
                # B index mapping: j, k. k = t*4 + k'
                for j in range(m):
                    for k_prime in range(4):
                        b_global = j * p + (t * 4 + k_prime)
                        b_local = j * 4 + k_prime
                        V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
                # C index mapping: k, i. k = t*4 + k'
                for k_prime in range(4):
                    for i in range(n):
                        d_global = (t * 4 + k_prime) * n + i
                        d_local = k_prime * n + i
                        W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
                        
            # Verify
            T_full = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_full[i*m+j, j*p+k, k*n+i] = 1.0
                        
            T_decomp = np.einsum('at,bt,dt->abd', U_full, V_full, W_full, optimize=True)
            if np.allclose(T_decomp, T_full, atol=1e-10, rtol=0):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback to rank 28
    U_full, V_full, W_full = _fallback_28()
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('at,bt,dt->abd', U_full, V_full, W_full, optimize=True)
    if not np.allclose(T_decomp, T_full, atol=1e-10, rtol=0):
        # Absolute fallback: schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
