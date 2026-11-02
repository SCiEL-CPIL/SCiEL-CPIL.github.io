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
    
    # Initialize with random values
    rank = 60  # try a rank less than 80
    np.random.seed(42)
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # Gradient descent
    lr = 0.01
    for step in range(10000):
        # Compute residuals
        # For non-zero entries
        prod_nonzero = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :], axis=1)
        residual_nonzero = prod_nonzero - 1.0
        
        # For zero entries, we need to compute sum_r U[a,r] V[b,r] W[c,r] for all (a,b,c) not in the nonzero set
        # This is expensive. Instead, we can compute the full tensor and subtract.
        
        # Full tensor: T_full[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 16x20x20 tensor, which is 6400 entries. We can compute it as:
        # T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor
        T_target = np.zeros((nm, mp, pn))
        T_target[idx_a, idx_b, idx_c] = 1.0
        
        # Residual
        residual = T_full - T_target
        loss = np.sum(residual**2)
        
        # Gradients
        # dL/dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
        dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
    
    # Snap to half-integers
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    
    # Verify
    T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
    error = np.sum((T_half - T_target)**2)
    
    if error < 1e-10:
        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    else:
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
