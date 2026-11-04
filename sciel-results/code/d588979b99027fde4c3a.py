<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10  # Very short search time to ensure safety
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Check exact decomposition
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                    if abs(val - T[a, b, c]) > 1e-9:
                        return False
        return True
=======
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s kill
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Vectorized check: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Reshape to (nm, mp, pn, rank) and sum over rank
        U_exp = U_r[:, np.newaxis, np.newaxis, :]  # (nm, 1, 1, rank)
        V_exp = V_r[np.newaxis, :, np.newaxis, :]  # (1, mp, 1, rank)
        W_exp = W_r[np.newaxis, np.newaxis, :, :]  # (1, 1, pn, rank)
        reconstructed = np.sum(U_exp * V_exp * W_exp, axis=-1)  # (nm, mp, pn)
        
        if not np.allclose(reconstructed, T, atol=tol):
            return False
        return True
>>>>>>> REPLACE

