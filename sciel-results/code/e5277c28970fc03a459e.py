import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Target Tensor indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    T_target = np.zeros((nm, mp, pn), dtype=np.complex128)
    T_target[idx_a, idx_b, idx_c] = 1.0 + 0.0j
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for safety
    
    # Try ranks from low to high
    # Lower bound is 49.
    ranks_to_try = list(range(49, 81))
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts depends on rank and time
        # Lower ranks are harder, need more restarts
        if rank < 60:
            num_restarts = 20
        else:
            num_restarts = 5
            
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(123 + restart * 1000 + rank)
            
            # Initialize with complex random values
            U = np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)
            V = np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)
            W = np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Gradient Descent
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            current_lr = lr
            patience = 0
            best_loss = np.inf
            max_steps = 3000
            
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_full - T_target
                loss = np.real(np.sum(residual**2))
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                    if patience > 50:
                        current_lr *= 0.9
                        patience = 0
                        if current_lr < 1e-6:
                            break
                
                dU = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
                dV = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
                dW = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
                
                vU = momentum * vU - current_lr * dU
                vV = momentum * vV - current_lr * dV
                vW = momentum * vW - current_lr * dW
                
                U += vU
                V += vV
                W += vW
                
            # Check for real half-integer solution
            U_imag = np.imag(U)
            V_imag = np.imag(V)
            W_imag = np.imag(W)
            
            if np.max(np.abs(U_imag)) > 0.05 or np.max(np.abs(V_imag)) > 0.05 or np.max(np.abs(W_imag)) > 0.05:
                continue
                
            U_real = np.real(U)
            V_real = np.real(V)
            W_real = np.real(W)
            
            U_half = np.round(U_real * 2) / 2
            V_half = np.round(V_real * 2) / 2
            W_half = np.round(W_real * 2) / 2
            
            T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
            error = np.max(np.abs(T_half - T_target))
            
            if error < 1e-9:
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }

    # Fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
