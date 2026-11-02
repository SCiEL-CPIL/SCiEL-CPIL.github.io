    def construct():
        n, m, p = 3, 4, 4
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Define the 11 terms for a 3x2x2 block
        # Each term is (u_list, v_list, w_list)
        # u_list: [(i, j, coeff), ...]
        # v_list: [(j, k, coeff), ...]
        # w_list: [(k, i, coeff), ...]
        
        terms = [
            # Strassen
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
            ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
            ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            ([(0,0,1), (1,0,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
            ([(1,1,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,1,1), (0,1,1)], [(0,0,1)]),
            # Trivial
            ([(2,0,1)], [(0,0,1)], [(0,2,1)]),
            ([(2,1,1)], [(1,0,1)], [(0,2,1)]),
            ([(2,0,1)], [(0,1,1)], [(1,2,1)]),
            ([(2,1,1)], [(1,1,1)], [(1,2,1)])
        ]
        
        blocks = [
            (0, 0), # j_base, k_base
            (0, 2),
            (2, 0),
            (2, 2)
        ]
        
        for j_base, k_base in blocks:
            for u_list, v_list, w_list in terms:
                for i, j, c in u_list:
                    U[i * 4 + (j_base + j)][col] += c
                for j, k, c in v_list:
                    V[(j_base + j) * 4 + (k_base + k)][col] += c
                for k, i, c in w_list:
                    W[(k_base + k) * 3 + i][col] += c
                col += 1
                
        return {"U": U, "V": V, "W": W}
    