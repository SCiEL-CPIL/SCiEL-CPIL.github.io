import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor M flattened
    M = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
                
    # Schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    
    start_time = time.time()
    deadline = 100.0 # 100s limit to be safe
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find rank 41 or 40
    for target_rank in [41, 40]:
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Few restarts
        for restart in range(20):
            if time.time() - start_time > deadline:
                break
                
            np.random.seed(42 + restart) # Reproducible
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Gradient Descent
            lr = 0.001
            prev_err = np.inf
            for step in range(200):
                if time.time() - start_time > deadline:
                    break
                
                # Compute T
                # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # Efficient: T = np.einsum('at,bt,ct->abc', U, V, W)
                T = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - M
                err = np.sum(E**2)
                
                if err < 1e-10:
                    found = True
                    break
                
                if err > prev_err:
                    lr *= 0.5
                else:
                    prev_err = err
                    if step % 10 == 0:
                        lr = min(lr * 1.1, 0.01)
                        
                # Gradients
                # dU[a,t] = 2 * sum_{b,c} E[a,b,c] V[b,t] W[c,t]
                # dV[b,t] = 2 * sum_{a,c} E[a,b,c] U[a,t] W[c,t]
                # dW[c,t] = 2 * sum_{a,b} E[a,b,c] U[a,t] V[b,t]
                
                # Use einsum for gradients
                # E is (nm, mp, pn)
                # V is (mp, R), W is (pn, R)
                # grad_U[a,t] = sum_{b,c} E[a,b,c] V[b,t] W[c,t]
                grad_U = np.einsum('abc,bt,ct->at', E, V, W)
                grad_V = np.einsum('abc,at,ct->bt', E, U, W)
                grad_W = np.einsum('abc,at,bt->ct', E, U, V)
                
                U -= 2 * lr * grad_U
                V -= 2 * lr * grad_V
                W -= 2 * lr * grad_W
                
            if found:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                T_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
                if np.allclose(T_r, M, atol=1e-9):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
                    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
