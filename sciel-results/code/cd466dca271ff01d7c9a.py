import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Build target entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries.append((u, v, w))
    
    # Try to find a lower rank using ALS
    def try_rank(r, max_iter=100):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            for u in range(nm):
                cons = [(v, w) for (tu, v, w) in target_entries if tu == u]
                if not cons:
                    U[u] = 0
                    continue
                B = np.array([[V[v, c] * W[w, c] for c in range(r)] for (v, w) in cons])
                rhs = np.ones(len(cons))
                if r > 0:
                    U[u] = np.linalg.lstsq(B, rhs, rcond=None)[0]
            
            # Update V
            for v in range(mp):
                cons = [(u, w) for (u, tv, w) in target_entries if tv == v]
                if not cons:
                    V[v] = 0
                    continue
                B = np.array([[U[u, c] * W[w, c] for c in range(r)] for (u, w) in cons])
                rhs = np.ones(len(cons))
                if r > 0:
                    V[v] = np.linalg.lstsq(B, rhs, rcond=None)[0]
            
            # Update W
            for w in range(pn):
                cons = [(u, v) for (u, v, tw) in target_entries if tw == w]
                if not cons:
                    W[w] = 0
                    continue
                B = np.array([[U[u, c] * V[v, c] for c in range(r)] for (u, v) in cons])
                rhs = np.ones(len(cons))
                if r > 0:
                    W[w] = np.linalg.lstsq(B, rhs, rcond=None)[0]
            
            # Check error
            err = 0
            for (u, v, w) in target_entries:
                val = np.dot(U[u], V[v] * W[w])
                err = max(err, abs(val - 1.0))
            if err < 1e-9:
                return (U, V, W)
        
        return None
    
    # Try ranks from 30 to 54
    for r in range(30, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break
        res = try_rank(r)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    r = schoolbook_rank
    U = [[0] * r for _ in range(nm)]
    V = [[0] * r for _ in range(mp)]
    W = [[0] * r for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
