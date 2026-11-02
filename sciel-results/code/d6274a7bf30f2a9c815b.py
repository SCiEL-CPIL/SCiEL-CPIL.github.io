import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    nm, mp, pn = n*m, m*p, p*n  # 4, 14, 14
    
    # Build valid terms
    valid_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_terms.append((a, b, c))
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for (a, b, c) in valid_terms:
            s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(s - 1.0) > tol:
                return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    i2 = c % n
                    k2 = c // n
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(s) > tol:
                        return False
        return True

    # Build rank-25 Strassen solution
    def make_strassen_solution():
        U = np.zeros((nm, 25), dtype=float)
        V = np.zeros((mp, 25), dtype=float)
        W = np.zeros((pn, 25), dtype=float)
        
        def add_strassen_block(col_start, k_start):
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        add_strassen_block(U, V, W, 0, 0)
        add_strassen_block(U, V, W, 7, 2)
        add_strassen_block(U, V, W, 14, 4)
        
        col_start = 21
        k = 6
        U[0, col_start+0] += 1; V[7*0 + k, col_start+0] += 1; W[2*k + 0, col_start+0] += 1
        U[1, col_start+1] += 1; V[7*1 + k, col_start+1] += 1; W[2*k + 0, col_start+1] += 1
        U[2, col_start+2] += 1; V[7*0 + k, col_start+2] += 1; W[2*k + 1, col_start+2] += 1
        U[3, col_start+3] += 1; V[7*1 + k, col_start+3] += 1; W[2*k + 1, col_start+3] += 1
        
        return U, V, W
    
    U25, V25, W25 = make_strassen_solution()
    
    if verify(U25, V25, W25):
        return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # Fallback: rank-28 schoolbook
    U = np.zeros((nm, 28), dtype=float)
    V = np.zeros((mp, 28), dtype=float)
    W = np.zeros((pn, 28), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
                idx += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
