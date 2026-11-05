import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    deadline = time.time() + 100
    
    # Try ranks from 1 to rank_school
    for rank in range(1, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(rank * 1000 + trial)
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            lr = 0.005
            for step in range(1000):
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = T - recon
                loss = np.sum(err**2)
                if loss < 1e-10:
                    found = True
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if step % 100 == 0:
                    lr *= 0.95
            
            if found:
                # Round to nearest 0.5
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify
                recon_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.max(np.abs(T - recon_round)) < 1e-9:
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = rank
                    break
                else:
                    # Use the unrounded version if it's close enough
                    if loss < 1e-6:
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        found = True
                        break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
