import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Create the sparse tensor indices
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
    
    # Pre-compute target tensor for verification (dense)
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0

    # Try different ranks
    # Schoolbook is 80. We try to find lower.
    for rank in range(60, 81):
        # Try multiple seeds
        for seed in range(5):
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Adam Optimizer
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            t = 0
            lr = 0.01
            
            # Optimization loop
            for step in range(1500):
                t += 1
                lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
                
                # Compute residual
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                
                residual = T_full.copy()
                residual[idx_a, idx_b, idx_c] -= 1.0
                
                # Gradients
                dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adam Update
                mU = beta1 * mU + (1 - beta1) * dU
                vU = beta2 * vU + (1 - beta2) * (dU ** 2)
                U -= lr_t * mU / (np.sqrt(vU) + epsilon)
                
                mV = beta1 * mV + (1 - beta1) * dV
                vV = beta2 * vV + (1 - beta2) * (dV ** 2)
                V -= lr_t * mV / (np.sqrt(vV) + epsilon)
                
                mW = beta1 * mW + (1 - beta1) * dW
                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                W -= lr_t * mW / (np.sqrt(vW) + epsilon)
                
                # Early stopping
                if step % 50 == 0:
                    loss = np.sum(residual**2)
                    if loss < 1e-5:
                        break
            
            # Snap to half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            # Verify
            T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
            error = np.sum((T_half - T_target)**2)
            
            if error < 1e-10:
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        # If we get here, this rank didn't work with these seeds
    
    # Fallback to schoolbook
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
