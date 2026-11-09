import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    def build(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]], 1
        
        if n == 1 and m == 1:
            # C_k = A_0 B_k
            U = [[1] * p]
            V = [[0] * p for _ in range(p)]
            for i in range(p):
                V[i][i] = 1
            W = [[0] * p for _ in range(p)]
            for i in range(p):
                W[i][i] = 1
            return U, V, W, p
        
        if n == 1 and p == 1:
            # C_0 = sum A_j B_j
            U = [[0] * m for _ in range(m)]
            for i in range(m):
                U[i][i] = 1
            V = [[0] * m for _ in range(m)]
            for i in range(m):
                V[i][i] = 1
            W = [[1] * m]
            return U, V, W, m
            
        if m == 1 and p == 1:
            # C_i = A_i B_0
            U = [[0] * n for _ in range(n)]
            for i in range(n):
                U[i][i] = 1
            V = [[1] * n]
            W = [[0] * n for _ in range(n)]
            for i in range(n):
                W[i][i] = 1
            return U, V, W, n

        # Recursive case: split dimensions in half
        n1, n2 = n // 2, n - n // 2
        m1, m2 = m // 2, m - m // 2
        p1, p2 = p // 2, p - p // 2
        
        # Get factors for sub-problem n1 x m1 x p1
        Us, Vs, Ws, Rs = build(n1, m1, p1)
        
        U = [[0] * (7 * Rs) for _ in range(n * m)]
        V = [[0] * (7 * Rs) for _ in range(m * p)]
        W = [[0] * (7 * Rs) for _ in range(p * n)]
        
        # Helper to add sub-factors to global factors
        def add_factors(U_global, V_global, W_global, U_sub, V_sub, W_sub, 
                        a_blocks, b_blocks, c_blocks, 
                        offset, n, m, p, n1, m1, p1):
            # a_blocks: list of (start_i, start_j, coeff)
            # b_blocks: list of (start_j, start_k, coeff)
            # c_blocks: list of (start_i, start_k, coeff)
            
            for r in range(Rs):
                col = offset + r
                # Process U
                for (si, sj, coeff) in a_blocks:
                    for i in range(n1):
                        for j in range(m1):
                            val = U_sub[i * m1 + j][r]
                            if val != 0:
                                U_global[(si + i) * m + (sj + j)][col] += val * coeff
                                
                # Process V
                for (sj, sk, coeff) in b_blocks:
                    for j in range(m1):
                        for k in range(p1):
                            val = V_sub[j * p1 + k][r]
                            if val != 0:
                                V_global[(sj + j) * p + (sk + k)][col] += val * coeff
                                
                # Process W
                for (si, sk, coeff) in c_blocks:
                    for i in range(n1):
                        for k in range(p1):
                            val = W_sub[k * n1 + i][r] # Note: W_sub indices are k*n1+i
                            if val != 0:
                                W_global[(sk + k) * n + (si + i)][col] += val * coeff

        # Define the 7 Strassen multiplications
        # M1 = (A11+A22)(B11+B22) -> C11(+), C22(+)
        a1 = [(0, 0, 1), (n1, m1, 1)]
        b1 = [(0, 0, 1), (m1, p1, 1)]
        c1 = [(0, 0, 1), (n1, p1, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a1, b1, c1, 0 * Rs, n, m, p, n1, m1, p1)
        
        # M2 = (A21+A22)B11 -> C21(+)
        a2 = [(n1, 0, 1), (n1, m1, 1)]
        b2 = [(0, 0, 1)]
        c2 = [(n1, 0, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a2, b2, c2, 1 * Rs, n, m, p, n1, m1, p1)
        
        # M3 = A11(B12-B22) -> C12(+)
        a3 = [(0, 0, 1)]
        b3 = [(0, p1, 1), (m1, p1, -1)]
        c3 = [(0, p1, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a3, b3, c3, 2 * Rs, n, m, p, n1, m1, p1)
        
        # M4 = A22(B21-B11) -> C11(+)
        a4 = [(n1, m1, 1)]
        b4 = [(m1, 0, 1), (0, 0, -1)]
        c4 = [(0, 0, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a4, b4, c4, 3 * Rs, n, m, p, n1, m1, p1)
        
        # M5 = (A11+A12)B22 -> C11(-)
        a5 = [(0, 0, 1), (0, m1, 1)]
        b5 = [(m1, p1, 1)]
        c5 = [(0, 0, -1)]
        add_factors(U, V, W, Us, Vs, Ws, a5, b5, c5, 4 * Rs, n, m, p, n1, m1, p1)
        
        # M6 = (A21-A11)(B11+B12) -> C22(+)
        a6 = [(n1, 0, 1), (0, 0, -1)]
        b6 = [(0, 0, 1), (0, p1, 1)]
        c6 = [(n1, p1, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a6, b6, c6, 5 * Rs, n, m, p, n1, m1, p1)
        
        # M7 = (A12-A22)(B21+B22) -> C11(+)
        a7 = [(0, m1, 1), (n1, m1, -1)]
        b7 = [(m1, 0, 1), (m1, p1, 1)]
        c7 = [(0, 0, 1)]
        add_factors(U, V, W, Us, Vs, Ws, a7, b7, c7, 6 * Rs, n, m, p, n1, m1, p1)
        
        return U, V, W, 7 * Rs

    U, V, W, rank = build(n, m, p)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Fallback to schoolbook if too slow (should not happen)
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

    return {"U": U, "V": V, "W": W}
