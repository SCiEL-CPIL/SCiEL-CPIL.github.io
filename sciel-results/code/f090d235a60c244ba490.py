import numpy as np
import time
import random

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0
    
    # Try to find a decomposition with rank < 28
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline:
            break
        if _find_decomposition(n, m, p, rank, deadline):
            U, V, W = _get_decomposition()
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 28
    return _strassen_28()

def _strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen 7-term decomposition for 2x2x2
    # M1 = (A22 - A11) * (B11 - B22)  -> contributes to C11, C22
    # M2 = (A11 + A22) * B22           -> contributes to C22
    # M3 = A11 * (B22 - B21)           -> contributes to C21
    # M4 = A22 * (B11 - B12)           -> contributes to C12
    # M5 = (A11 + A12) * B12           -> contributes to C12
    # M6 = (A21 + A22) * B11           -> contributes to C21
    # M7 = (A11 - A21) * (B21 + B22)   -> contributes to C11
    
    # For 2x2 x 2x8, we have 4 column blocks of B (each 2x2)
    # Apply Strassen independently to each block -> rank 28
    
    u_coeffs = [
        # M1: A22 - A11
        [1, 0, 0, 1], 
        # M2: A11 + A22
        [0, 0, 1, 1], 
        # M3: A11
        [1, 0, 0, 0], 
        # M4: A22
        [0, 0, 0, 1], 
        # M5: A11 + A12
        [1, 1, 0, 0], 
        # M6: A21 + A22
        [-1, 0, 1, 0],
        # M7: A11 - A21
        [0, 1, 0, -1] 
    ]
    
    for k in range(4):
        base_v = 2 * k
        e_idx = base_v      # B11 of block k
        f_idx = base_v + 1  # B21 of block k
        g_idx = 8 + base_v  # B12 of block k
        h_idx = 9 + base_v  # B22 of block k
        
        base_w = 4 * k
        c11_idx = base_w
        c21_idx = base_w + 1
        c12_idx = base_w + 2
        c22_idx = base_w + 3
        
        for t in range(7):
            col = k * 7 + t
            
            for u_idx, val in enumerate(u_coeffs[t]):
                if val != 0:
                    U[u_idx][col] = val
            
            if t == 0:
                V[e_idx][col] = 1
                V[h_idx][col] = 1
            elif t == 1:
                V[e_idx][col] = 1
            elif t == 2:
                V[f_idx][col] = 1
                V[h_idx][col] = -1
            elif t == 3:
                V[g_idx][col] = 1
                V[e_idx][col] = -1
            elif t == 4:
                V[h_idx][col] = 1
            elif t == 5:
                V[e_idx][col] = 1
                V[f_idx][col] = 1
            elif t == 6:
                V[g_idx][col] = 1
                V[h_idx][col] = 1
            
            if t == 0:
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 1:
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
            elif t == 2:
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 3:
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
            elif t == 4:
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
            elif t == 5:
                W[c22_idx][col] = 1
            elif t == 6:
                W[c11_idx][col] = 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Reconstruct tensor
    # T[i, j, k] = sum_t U[t, i*m+j] * V[t, j*p+k] * W[t, k*n+i]
    # Check against target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                val = 0.0
                for t in range(rank):
                    val += U[t, i*m+j] * V[t, j*p+k] * W[t, k*n+i]
                if abs(val - target) > 1e-5:
                    return False
    return True

def _find_decomposition(n, m, p, rank, deadline):
    """
    Try to find a rank-rank decomposition using structured random search.
    Uses the tensor product structure: the 2x2x8 tensor is related to
    the 2x2x2 tensor (rank 7) tensor 4 (identity).
    
    Strategy: Start from the rank-28 Strassen decomposition and try to
    merge terms by finding linear dependencies.
    """
    start_time = time.time()
    
    # Get the rank-28 Strassen decomposition as starting point
    strassen = _strassen_28()
    U0 = np.array(strassen['U'], dtype=float)
    V0 = np.array(strassen['V'], dtype=float)
    W0 = np.array(strassen['W'], dtype=float)
    
    # Build the full tensor T[i,j,k] = sum_t U[t,i*m+j] * V[t,j*p+k] * W[t,k*n+i]
    # Target: T[i,j,k] = 1 for all i,j,k (since we're multiplying identity-like)
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor
    # in the appropriate sense - actually it's the structural tensor)
    
    # Let's use a different approach: gradient-based optimization
    # Minimize ||T_decomposed - T_target||^2
    
    # Target tensor
    T_target = np.ones((n, m, p))
    
    # Initialize with perturbed Strassen
    best_U, best_V, best_W = None, None, None
    best_loss = np.inf
    
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline - 5:
            break
        
        # Initialize from Strassen with random perturbations
        scale = 0.1 + 0.5 * (attempt / n_attempts)
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Add random noise
        U += scale * np.random.randn(n*m, rank)
        V += scale * np.random.randn(m*p, rank)
        W += scale * np.random.randn(p*n, rank)
        
        # Gradient descent
        lr = 0.01
        n_steps = 200
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor
            T = np.zeros((n, m, p))
            for t in range(rank):
                T += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
            
            # Compute loss
            diff = T - T_target
            loss = np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-6:
                break
            
            # Compute gradients
            # dL/dU[i,t] = 2 * sum_{j,k} diff[i,j,k] * V[j,t] * W[k,t]
            # dL/dV[j,t] = 2 * sum_{i,k} diff[i,j,k] * U[i,t] * W[k,t]
            # dL/dW[k,t] = 2 * sum_{i,j} diff[i,j,k] * U[i,t] * V[j,t]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for t in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i*m+j, t] += 2 * diff[i,j,k] * V[j*p+k, t] * W[k*n+i, t]
                            dV[j*p+k, t] += 2 * diff[i,j,k] * U[i*m+j, t] * W[k*n+i, t]
                            dW[k*n+i, t] += 2 * diff[i,j,k] * U[i*m+j, t] * V[j*p+k, t]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Try to snap to integers/half-integers
        if best_loss < 0.01:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            # Verify snapped version
            T_snap = np.zeros((n, m, p))
            for t in range(rank):
                T_snap += np.einsum('i,j,k->ijk', U_snap[:, t], V_snap[:, t], W_snap[:, t])
            
            if np.allclose(T_snap, T_target, atol=1e-3):
                _store_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                return True
    
    # Check if we found a good solution
    if best_U is not None and best_loss < 1e-4:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        T_snap = np.zeros((n, m, p))
        for t in range(rank):
            T_snap += np.einsum('i,j,k->ijk', U_snap[:, t], V_snap[:, t], W_snap[:, t])
        
        if np.allclose(T_snap, T_target, atol=1e-3):
            _store_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            return True
    
    return False

_decomposition = [None, None, None]

def _store_decomposition(U, V, W):
    _decomposition[0] = U
    _decomposition[1] = V
    _decomposition[2] = W

def _get_decomposition():
    return _decomposition[0], _decomposition[1], _decomposition[2]
