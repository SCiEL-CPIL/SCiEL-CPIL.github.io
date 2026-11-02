import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0
    
    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20
    
    def build_strassen_rank35():
        rank = 35
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        # Strassen 7-term decomposition for 2x2x2
        strassen = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
            ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        col = 0
        for block in range(5):
            for a, b, c in strassen:
                U[0, col] = a[0]
                U[1, col] = a[1]
                U[2, col] = a[2]
                U[3, col] = a[3]
                
                V[0*p + 2*block, col] = b[0]
                V[0*p + 2*block+1, col] = b[1]
                V[1*p + 2*block, col] = b[2]
                V[1*p + 2*block+1, col] = b[3]
                
                W[(2*block)*2 + 0, col] = c[0]
                W[(2*block)*2 + 1, col] = c[1]
                W[(2*block+1)*2 + 0, col] = c[2]
                W[(2*block+1)*2 + 1, col] = c[3]
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = build_strassen_rank35()
    
    return best
