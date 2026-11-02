import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    deadline = time.time() + 140 # Safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        M = np.einsum('ar,cr->acr', U, W)
        T_approx = np.einsum('acr,br->abc', M, V)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
        
    def try_round_and_verify(U, V, W):
        """Try rounding to various fractions and verify"""
        scales = [1, 2, 4, 8]
        for s in scales:
            U_r = np.round(U * s) / s
            V_r = np.round(V * s) / s
            W_r = np.round(W * s) / s
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None
        
    def optimize_rank(target_rank, max_restarts=30, max_steps=3000):
        """Try to find a decomposition of given rank"""
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
                
            # Initialization
            np.random.seed(1000 + restart)
            if restart % 5 == 0:
                # Random
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            elif restart % 5 == 1:
                # From schoolbook (subset)
                idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                U = U_sb[:, idx].copy()
                V = V_sb[:, idx].copy()
                W = W_sb[:, idx].copy()
            elif restart % 5 == 2:
                # Sparse random
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
                # Add schoolbook structure
                idx = np.random.choice(rank_sb, size=min(5, target_rank), replace=False)
                U[:, :len(idx)] += U_sb[:, idx]
                V[:, :len(idx)] += V_sb[:, idx]
                W[:, :len(idx)] += W_sb[:, idx]
            elif restart % 5 == 3:
                # Perturbed schoolbook
                idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                U = U_sb[:, idx].copy()
                V = V_sb[:, idx].copy()
                W = W_sb[:, idx].copy()
                U += np.random.randn(nm, target_rank) * 0.1
                V += np.random.randn(mp, target_rank) * 0.1
                W += np.random.randn(pn, target_rank) * 0.1
            else:
                # Different random scale
                U = np.random.randn(nm, target_rank) * 1.0
                V = np.random.randn(mp, target_rank) * 1.0
                W = np.random.randn(pn, target_rank) * 1.0
                
            # Adam Optimizer
            lr = 0.05
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            t = 0
            best_res = np.inf
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                t += 1
                
                # Forward
                M = np.einsum('ar,cr->acr', U, W)
                T_approx = np.einsum('acr,br->abc', M, V)
                R = T - T_approx
                res = np.sum(R**2)
                
                if res < best_res:
                    best_res = res
                    if best_res < 1e-10:
                        break
                        
                # Backward (Gradients)
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                
                # Adam Update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * (grad_U ** 2)
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Decay learning rate
                if step > 1000:
                    lr *= 0.999
                    
            if best_res < 1e-8:
                rounded = try_round_and_verify(U, V, W)
                if rounded is not None:
                    return rounded
                elif best_res < 1e-10:
                    return U, V, W
                    
        return None

    # Try ranks from 49 down to 40
    for target_rank in range(rank_sb - 1, 39, -1):
        if time.time() > deadline - 20:
            break
            
        result = optimize_rank(target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = target_rank
                continue
        else:
            break
            
    return best_result
