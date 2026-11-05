    DIMENSIONS = [2, 5, 5]

    def construct():
        n, m, p = DIMENSIONS
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_222(j0, k0):
            nonlocal col
            # M1
            U[0*m + j0][col] = 1
            U[1*m + j0+1][col] = 1
            V[j0*p + k0][col] = 1
            V[(j0+1)*p + k0+1][col] = 1
            W[k0*n + 0][col] = 1
            W[(k0+1)*n + 1][col] = 1
            col += 1
            # M2
            U[1*m + j0][col] = 1
            U[1*m + j0+1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            W[(k0+1)*n + 1][col] = -1
            col += 1
            # M3
            U[0*m + j0][col] = 1
            V[(j0+1)*p + k0][col] = 1
            V[(j0+1)*p + k0+1][col] = -1
            W[(k0+1)*n + 0][col] = 1
            W[(k0+1)*n + 1][col] = 1
            col += 1
            # M4
            U[1*m + j0+1][col] = 1
            V[j0*p + k0+1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + 0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1
            # M5
            U[0*m + j0][col] = 1
            U[0*m + j0+1][col] = 1
            V[(j0+1)*p + k0+1][col] = 1
            W[k0*n + 0][col] = -1
            W[(k0+1)*n + 0][col] = 1
            col += 1
            # M6
            U[1*m + j0][col] = 1
            U[0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j0*p + k0+1][col] = 1
            W[(k0+1)*n + 1][col] = 1
            col += 1
            # M7
            U[0*m + j0+1][col] = 1
            U[1*m + j0+1][col] = -1
            V[(j0+1)*p + k0][col] = 1
            V[(j0+1)*p + k0+1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1

        def add_221(j0, k0):
            # A 2x2, B 2x1, C 2x1
            # c0 = a00 b0 + a01 b1
            # c1 = a10 b0 + a11 b1
            nonlocal col
            # T1: a00 b0 -> c0
            U[0*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
            # T2: a01 b1 -> c0
            U[0*m + j0+1][col] = 1
            V[(j0+1)*p + k0][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
            # T3: a10 b0 -> c1
            U[1*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1
            # T4: a11 b1 -> c1
            U[1*m + j0+1][col] = 1
            V[(j0+1)*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1

        def add_212(j0, k0):
            # A 2x1, B 1x2, C 2x2
            # c00 = a0 b0
            # c01 = a0 b1
            # c10 = a1 b0
            # c11 = a1 b1
            nonlocal col
            # T1: a0 b0 -> c00
            U[0*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
            # T2: a0 b1 -> c01
            U[0*m + j0][col] = 1
            V[j0*p + k0+1][col] = 1
            W[(k0+1)*n + 0][col] = 1
            col += 1
            # T3: a1 b0 -> c10
            U[1*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1
            # T4: a1 b1 -> c11
            U[1*m + j0][col] = 1
            V[j0*p + k0+1][col] = 1
            W[(k0+1)*n + 1][col] = 1
            col += 1

        def add_211(j0, k0):
            # A 2x1, B 1x1, C 2x1
            nonlocal col
            # T1: a0 b -> c0
            U[0*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
            # T2: a1 b -> c1
            U[1*m + j0][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 1][col] = 1
            col += 1

        # m blocks: 0-1, 2-3, 4
        # p blocks: 0-1, 2-3, 4
        
        # (0,0) -> 222
        add_strassen_222(0, 0)
        # (0,2) -> 222
        add_strassen_222(0, 2)
        # (0,4) -> 221
        add_221(0, 4)
        
        # (2,0) -> 222
        add_strassen_222(2, 0)
        # (2,2) -> 222
        add_strassen_222(2, 2)
        # (2,4) -> 221
        add_221(2, 4)
        
        # (4,0) -> 212
        add_212(4, 0)
        # (4,2) -> 212
        add_212(4, 2)
        # (4,4) -> 211
        add_211(4, 4)
        
        return {"U": U, "V": V, "W": W}
    