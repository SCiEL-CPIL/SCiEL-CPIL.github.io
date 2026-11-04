"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill
    
    rank_schoolbook = n * m * p  # 30
    
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check valid terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        # Check invalid terms
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                    if abs(total - expected) > 1e-9:
                        return False
        return True
    
    U0, V0, W0 = make_schoolbook(n, m, p)
    
    # Build the target tensor T[a,b,c]
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def try_rank(r, n, m, p, T, time_limit):
        """Try to find a rank-r decomposition using vectorized Adam optimizer."""
        nm, mp, pn = n*m, m*p, p*n
        
        for attempt in range(50):
            if time.time() > time_limit:
                return None
            
            np.random.seed(attempt * 12345 + 67890)
            # Initialize with small random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t_step = 0
            lr = 0.05
            
            for iteration in range(1000):
                if time.time() > time_limit:
                    break
                t_step += 1
                
                C = np.einsum('ar,br,cr->abc', U, V, W)
                err = T - C
                
                grad_U = -2 * np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = -2 * np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = -2 * np.einsum('abc,ar,br->cr', err, U, V)
                
                m_U = beta1*m_U + (1-beta1)*grad_U
                v_U = beta2*v_U + (1-beta2)*grad_U**2
                m_U_hat = m_U / (1 - beta1**t_step)
                v_U_hat = v_U / (1 - beta2**t_step)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1*m_V + (1-beta1)*grad_V
                v_V = beta2*v_V + (1-beta2)*grad_V**2
                m_V_hat = m_V / (1 - beta1**t_step)
                v_V_hat = v_V / (1 - beta2**t_step)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1*m_W + (1-beta1)*grad_W
                v_W = beta2*v_W + (1-beta2)*grad_W**2
                m_W_hat = m_W / (1 - beta1**t_step)
                v_W_hat = v_W / (1 - beta2**t_step)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if iteration % 100 == 0:
                    loss = np.sum(err**2)
                    if loss < 1e-8:
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        U_list = U_round.tolist()
                        V_list = V_round.tolist()
                        W_list = W_round.tolist()
                        if verify(U_list, V_list, W_list, n, m, p):
                            return U_list, V_list, W_list
                        else:
                            U_list = U.tolist()
                            V_list = V.tolist()
                            W_list = W.tolist()
                            if verify(U_list, V_list, W_list, n, m, p):
                                return U_list, V_list, W_list
            
            C = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T - C)**2)
            if loss < 1e-6:
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                if verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
                else:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 29 down to 20
    for r in range(rank_schoolbook - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, n, m, p, T, deadline - 5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
