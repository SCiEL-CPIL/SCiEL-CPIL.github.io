import numpy as np
import scipy.optimize as opt

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Target tensor indices
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
    
    # Try ranks from low to high
    # Known bounds suggest ~60-70. Let's try 64, 65, 66, 68, 70, 72, 75, 80
    ranks_to_try = [64, 65, 66, 68, 70, 72, 75, 78, 80]
    
    for rank in ranks_to_try:
        # Try a few random seeds
        for seed in range(10):
            np.random.seed(seed)
            # Initialize
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss_and_grad(x):
                U_flat, V_flat, W_flat = x[:nm*rank], x[nm*rank:nm*rank+mp*rank], x[nm*rank+mp*rank:]
                U = U_flat.reshape(nm, rank)
                V = V_flat.reshape(mp, rank)
                W = W_flat.reshape(pn, rank)
                
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_full - T_target
                loss = np.sum(residual**2)
                
                # Gradients
                dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            try:
                # Use L-BFGS-B for robustness
                res = opt.minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'maxfun': 5000})
                
                # Extract optimized matrices
                U_opt, V_opt, W_opt = res.x[:nm*rank], res.x[nm*rank:nm*rank+mp*rank], res.x[nm*rank+mp*rank:]
                U_opt = U_opt.reshape(nm, rank)
                V_opt = V_opt.reshape(mp, rank)
                W_opt = W_opt.reshape(pn, rank)
                
                # Snap to half-integers
                U_half = np.round(U_opt * 2) / 2
                V_half = np.round(V_opt * 2) / 2
                W_half = np.round(W_opt * 2) / 2
                
                # Verify
                T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                error = np.sum((T_half - T_target)**2)
                
                if error < 1e-10:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
            except Exception as e:
                continue

    # Fallback to schoolbook rank 80
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
