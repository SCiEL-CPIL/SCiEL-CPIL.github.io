import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # safety margin
    
    # Precompute the target tensor T
    # T[a,b,c] = 1 if a%m == b//p and c == (b%p)*n + a//m, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    
    # Try ranks from 12 upwards (12 is a strong candidate for 2x3x3)
    # Schoolbook is 18.
    for rank in range(12, 19):
        if time.time() > deadline - 10:
            break
        
        # Number of trials: more for lower ranks
        trials = 50 if rank <= 13 else 20
        
        for trial in range(trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                
                # Solve for U
                # T[a] = sum_s U[a,s] * (V[:,s] * W[:,s])
                M_U = np.einsum('bs,cs->bcs', V, W)
                M_U_mat = M_U.transpose(2, 0, 1).reshape(rank, -1)
                T_mat = T.reshape(n*m, -1)
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U_mat.T, T_mat.T, rcond=None)
                    U = U_new.T
                except Exception:
                    pass
                
                # Solve for V
                # T[b] = sum_s V[b,s] * (U[:,s] * W[:,s])
                Q_V = np.einsum('as,cs->acs', U, W)
                Q_V_mat = Q_V.transpose(2, 0, 1).reshape(rank, -1)
                T_V_mat = T.transpose(1, 0, 2).reshape(m*p, -1)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(Q_V_mat.T, T_V_mat.T, rcond=None)
                    V = V_new.T
                except Exception:
                    pass
                
                # Solve for W
                # T[c] = sum_s W[c,s] * (U[:,s] * V[:,s])
                Q_W = np.einsum('as,bs->abs', U, V)
                Q_W_mat = Q_W.transpose(2, 0, 1).reshape(rank, -1)
                T_W_mat = T.transpose(2, 0, 1).reshape(p*n, -1)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(Q_W_mat.T, T_W_mat.T, rcond=None)
                    W = W_new.T
                except Exception:
                    pass
            
            # Check residual
            R = np.einsum('as,bs,cs->abc', U, V, W)
            cost = np.sum((R - T)**2)
            
            if cost < 1e-9:
                # Try to round to half-integers
                U_out = np.round(U * 2) / 2
                V_out = np.round(V * 2) / 2
                W_out = np.round(W * 2) / 2
                
                R_check = np.einsum('as,bs,cs->abc', U_out, V_out, W_out)
                if np.allclose(R_check, T, atol=1e-9):
                    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        
    # Fallback: schoolbook
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
