"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Try rank 45: tile (2,3,3) rank-15 decomposition 3 times
    # First, find (2,3,3) rank-15 decomposition
    
    nn, mm, pp = 2, 3, 3
    target_rank = 15
    
    best_solution = None
    np.random.seed(123)
    
    for trial in range(100):
        if time.time() > deadline - 20:
            break
            
        U = np.random.randn(nn*mm, target_rank) * 0.5
        V = np.random.randn(mm*pp, target_rank) * 0.5
        W = np.random.randn(pp*nn, target_rank) * 0.5
        
        # Adam-like optimizer
        lr = 0.05
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        
        for step in range(500):
            if time.time() > deadline - 20:
                break
                
            t = step + 1
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            err = 0
            
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = 0
                        for r in range(target_rank):
                            val += U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r]
                        diff = val - 1
                        err += diff * diff
                        for r in range(target_rank):
                            grad_U[i*mm+j, r] += 2*diff * V[j*pp+k, r] * W[k*nn+i, r]
                            grad_V[j*pp+k, r] += 2*diff * U[i*mm+j, r] * W[k*nn+i, r]
                            grad_W[k*nn+i, r] += 2*diff * U[i*mm+j, r] * V[j*pp+k, r]
            
            # Adam update
            m_U = b1*m_U + (1-b1)*grad_U
            v_U = b2*v_U + (1-b2)*grad_U**2
            m_U_hat = m_U/(1-b1**t)
            v_U_hat = v_U/(1-b2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = b1*m_V + (1-b1)*grad_V
            v_V = b2*v_V + (1-b2)*grad_V**2
            m_V_hat = m_V/(1-b1**t)
            v_V_hat = v_V/(1-b2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = b1*m_W + (1-b1)*grad_W
            v_W = b2*v_W + (1-b2)*grad_W**2
            m_W_hat = m_W/(1-b1**t)
            v_W_hat = v_W/(1-b2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if err < 1e-12:
                break
        
        # Check if solution is good
        final_err = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    val = 0
                    for r in range(target_rank):
                        val += U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r]
                    final_err += (val - 1)**2
        
        if final_err < 1e-8 and best_solution is None:
            best_solution = (U.copy(), V.copy(), W.copy(), final_err)
            break
    
    if best_solution is not None:
        U233, V233, W233, err = best_solution
        rank = 45
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        
        for g in range(3):
            offset = g * 15
            U[:, offset:offset+15] = U233
            for j in range(3):
                for k in range(3):
                    V[j*9 + g*3 + k, offset:offset+15] = V233[j*3 + k, :]
            for k in range(3):
                for i in range(2):
                    W[(g*3+k)*2 + i, offset:offset+15] = W233[k*2 + i, :]
        
        def to_list(matrix):
            rounded = np.round(matrix * 2) / 2
            return [[float(x) for x in row] for row in rounded]
        
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
