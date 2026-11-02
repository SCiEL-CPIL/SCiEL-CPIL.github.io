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
    
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try a range of ranks, starting from low to high
    # Known bounds for M_4,4,4 suggest rank <= 28 or 49? 
    # For M_4,4,5, it's likely lower than 80.
    # Let's try ranks 32, 40, 48, 56, 64, 72, 79
    candidate_ranks = [32, 40, 48, 56, 64, 72, 79]
    
    # If none work, fallback to schoolbook
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

    for rank in candidate_ranks:
        np.random.seed(42 + rank)
        # Initialize with complex values
        U = np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)
        V = np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)
        W = np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)
        
        # Gradient Descent
        lr = 0.01
        for step in range(2000):
            # Compute residual
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T_full - T_target
            loss = np.sum(residual**2).real
            
            # Gradients (conjugate gradient for complex)
            # dL/dU = 2 * residual * conj(V) * conj(W)
            dU = 2 * np.einsum('abc,br,cr->ar', residual, np.conj(V), np.conj(W))
            dV = 2 * np.einsum('abc,ar,cr->br', residual, np.conj(U), np.conj(W))
            dW = 2 * np.einsum('abc,ar,br->cr', residual, np.conj(U), np.conj(V))
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Early stopping if loss is very small
            if loss < 1e-8:
                break
        
        # Check if imaginary parts are negligible
        if np.max(np.abs(U.imag)) > 1e-4 or np.max(np.abs(V.imag)) > 1e-4 or np.max(np.abs(W.imag)) > 1e-4:
            continue
            
        # Snap to half-integers
        U_real = U.real
        V_real = V.real
        W_real = W.real
        
        U_half = np.round(U_real * 2) / 2
        V_half = np.round(V_real * 2) / 2
        W_half = np.round(W_real * 2) / 2
        
        # Verify exact decomposition with snapped values
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10:
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
