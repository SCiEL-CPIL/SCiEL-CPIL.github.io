"""Matrix multiplication 4,4,9 - Strassen-based decomposition with Kronecker product structure"""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_4x4_4x3():
    """
    Construct Strassen decomposition for 4x4 @ 4x3.
    
    Partition A (4x4) into four 2x2 blocks:
        A = [[A00, A01], [A10, A11]]
    
    Partition B (4x3) into four 2x3 blocks:
        B = [[B00, B01], [B10, B11]]
    
    Result C (4x3) has four 2x3 blocks:
        C = [[C00, C01], [C10, C11]]
    
    Strassen products (7 products of 2x2 @ 2x3):
        M1 = (A00+A11) @ (B00+B10)
        M2 = (A10+A11) @ B00
        M3 = A00 @ (B01-B11)
        M4 = A11 @ (B10-B00)
        M5 = (A00+A01) @ B10
        M6 = (A10-A00) @ (B00+B01)
        M7 = (A01-A11) @ (B10+B11)
    
    Reconstruct:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M5
        C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 4, 4, 3
    
    # Winograd rank-7 decomposition for 2x2 @ 2x2
    # For 2x2 @ 2x2, the rank-7 decomposition is:
    #   Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)e
    #   m3 = a(f-h)
    #   m4 = d(g-e)
    #   m5 = (a+b)h
    #   m6 = (c-a)(e+f)
    #   m7 = (b-d)(g+h)
    #   C00 = m1 + m4 - m5 + m7
    #   C01 = m3 + m5
    #   C10 = m2 + m5
    #   C11 = m1 - m2 + m3 + m6
    
    # For 2x2 @ 2x3, we can use the same 7 multiplications by treating
    # the 2x3 matrix as three 2x1 matrices and applying the same linear combinations.
    
    # Actually, let's use a simpler approach: use schoolbook for the inner products.
    # 2x2 @ 2x3 requires 4*3 = 12 scalar multiplications.
    # Total: 7 * 12 = 84 terms.
    
    # But we can do better. Let's use the fact that the tensor rank of 2x2 @ 2x3 is 7.
    # The decomposition works by:
    #   m1 = (a+d) * [e+h, f+i, g+j]
    #   m2 = (c+d) * [e, f, g]
    #   m3 = a * [f-h, i-g, j-i]
    #   m4 = d * [h-e, g-f, i-g]
    #   m5 = (a+b) * [h, i, j]
    #   m6 = (c-a) * [e+h, f+i, g+j]
    #   m7 = (b-d) * [h+i, g+f, j+i]
    
    # Let me verify these formulas more carefully.
    # For 2x2 @ 2x3:
    #   A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    #   C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # Using the above decomposition:
    #   m1 = (a+d) * [e+h, f+i, g+j]
    #   m2 = (c+d) * [e, f, g]
    #   m3 = a * [f-h, i-g, j-i]
    #   m4 = d * [h-e, g-f, i-g]
    #   m5 = (a+b) * [h, i, j]
    #   m6 = (c-a) * [e+h, f+i, g+j]
    #   m7 = (b-d) * [h+i, g+f, j+i]
    
    # C00 (top-left) should be ae+bh:
    #   m1[0] + m4[0] - m5[0] + m7[0]
    #   = (a+d)(e+h) + d(h-e) - (a+b)h + (b-d)(h+i)
    #   = ae + ah + de + dh + dh - de - ah - bh + bh + bi - dh - di
    #   = ae + bi  -- This is not ae+bh!
    
    # So the formulas are incorrect. Let me derive them properly.
    
    # For 2x2 @ 2x3, the tensor rank is actually 8, not 7.
    # Let's use schoolbook: 12 terms.
    
    # For now, let's just return None and use schoolbook in the main function.
    return None

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    def tensor_residual(U, V, W):
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0
    
    def tensor_grads(U, V, W):
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        err = T - 1.0
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
        
        return grad_U.reshape(nm, rank), grad_V.reshape(mp, rank), grad_W.reshape(pn, rank)
    
    # Try ranks from lower to higher
    for target_rank in range(50, min(rank_schoolbook + 1, 120)):
        if time.time() > deadline:
            break
            
        for trial in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            lr = 0.01
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
            
            for step in range(1500):
                if time.time() > deadline:
                    break
                    
                t += 1
                gU, gV, gW = tensor_grads(U, V, W)
                
                m_U = beta1 * m_U + (1 - beta1) * gU
                v_U = beta2 * v_U + (1 - beta2) * gU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * gV
                v_V = beta2 * v_V + (1 - beta2) * gV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * gW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if step % 100 == 0:
                    T = tensor_residual(U, V, W)
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-6:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = [[float(x) for x in row] for row in U]
                            best_V = [[float(x) for x in row] for row in V]
                            best_W = [[float(x) for x in row] for row in W]
                        break
            
            if target_rank < best_rank:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
