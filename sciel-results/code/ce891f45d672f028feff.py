import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    def _from_columns(cols, n, m, p, rank):
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r, (Ac, Bc, Cc) in enumerate(cols):
            for (i, j), v in Ac.items():
                U[i*m+j][r] += v
            for (j, k), v in Bc.items():
                V[j*p+k][r] += v
            for (i, k), v in Cc.items():
                W[k*n+i][r] += v
        return U, V, W

    def _verify(U, V, W, n, m, p, rank):
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    key = (i*m+j, j*p+k, k*n+i)
                    T[key] = T.get(key, 0) + 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - T.get((a, b, c), 0)) > 1e-9:
                        return False
        return True

    def _build_rank_34(n, m, p):
        cols = []
        
        def strassen(ai, aj, bj, bk, ci, ck):
            def A(i, j): return (ai+i, aj+j)
            def B(j, k): return (bj+j, bk+k)
            def C(i, k): return (ci+i, ck+k)
            M = [
                ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
                ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
                ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
                ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
                ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
                ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
                ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
            ]
            for Ac, Bc, Cc in M:
                cols.append((dict(Ac), dict(Bc), dict(Cc)))
        
        # 1. Strassen A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        strassen(0, 0, 0, 0, 0, 0)
        
        # 2. Strassen A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        strassen(0, 0, 0, 2, 0, 2)
        
        # 3. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # 4 terms
        for i in range(2):
            for k in range(2):
                cols.append((
                    {(i, 2): 1},
                    {(2, k): 1},
                    {(i, k): 1}
                ))
        
        # 4. A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
        # 4 terms
        for i in range(2):
            for k in range(2):
                cols.append((
                    {(i, 2): 1},
                    {(2, 2+k): 1},
                    {(i, 2+k): 1}
                ))
        
        # 5. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        # 4 terms
        for j in range(2):
            for k in range(2):
                cols.append((
                    {(2, j): 1},
                    {(j, k): 1},
                    {(2, k): 1}
                ))
        
        # 6. A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        # 4 terms
        for j in range(2):
            for k in range(2):
                cols.append((
                    {(2, j): 1},
                    {(j, 2+k): 1},
                    {(2, 2+k): 1}
                ))
        
        # 7. A[2, 2] * B[2, 0:2] -> C[2, 0:2]
        # 2 terms
        for k in range(2):
            cols.append((
                {(2, 2): 1},
                {(2, k): 1},
                {(2, k): 1}
            ))
        
        # 8. A[2, 2] * B[2, 2:4] -> C[2, 2:4]
        # 2 terms
        for k in range(2):
            cols.append((
                {(2, 2): 1},
                {(2, 2+k): 1},
                {(2, 2+k): 1}
            ))
        
        rank = len(cols)
        return _from_columns(cols, n, m, p, rank)

    U, V, W = _build_rank_34(n, m, p)
    rank_34 = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank_34):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
