import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    rank = 81
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    strassen_terms = [
        ({0:1,3:1}, {0:1,3:1}, {0:1,3:1}),
        ({2:1,3:1}, {0:1}, {1:1,3:1}),
        ({0:1}, {1:1,3:-1}, {2:1,3:1}),
        ({3:1}, {2:1,0:-1}, {0:1,1:1}),
        ({0:1,1:1}, {3:1}, {0:1,2:1}),
        ({2:1,0:-1}, {0:1,1:1}, {3:1}),
        ({1:1,3:-1}, {2:1,3:1}, {0:1}),
    ]
    
    def add_strassen_sub(rank, a_blocks, b_blocks, c_blocks):
        for u_local, v_local, w_local in strassen_terms:
            for (ai, aj, ac) in a_blocks:
                for li, lc in u_local.items():
                    r = li // 2
                    c = li % 2
                    U[(ai*2+r)*4 + (aj*2+c)][rank] += ac * lc
            for (bi, bj, bc) in b_blocks:
                for li, lc in v_local.items():
                    r = li // 2
                    c = li % 2
                    V[(bi*2+r)*4 + (bj*2+c)][rank] += bc * lc
            for (ci, cj, cc) in c_blocks:
                for li, lc in w_local.items():
                    k_local = li // 2
                    i_local = li % 2
                    W[(cj*2+k_local)*4 + (ci*2+i_local)][rank] += cc * lc
            rank += 1
        return rank
    
    rank = add_strassen_sub(rank, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    rank = add_strassen_sub(rank, [(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)])
    rank = add_strassen_sub(rank, [(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
    rank = add_strassen_sub(rank, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
    rank = add_strassen_sub(rank, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)])
    rank = add_strassen_sub(rank, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    rank = add_strassen_sub(rank, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*4 + j][rank] = 1
                V[j*4 + k][rank] = 1
                W[k*4 + i][rank] = 1
                rank += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(6):
                val = np.sum(U_np[i*4+j] * V_np[j*4+k] * W_np[k*4+i])
                if abs(val - 1) > 1e-9:
                    raise ValueError(f"Decomposition error at ({i},{j},{k}): got {val}")
    
    return {"U": U, "V": V, "W": W}
