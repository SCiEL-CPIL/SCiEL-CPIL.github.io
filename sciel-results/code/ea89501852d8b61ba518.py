def generate_strassen_vectors(n):
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    U_sub, V_sub, W_sub = generate_strassen_vectors(n // 2)
    rank_sub = len(U_sub)
    
    U = []
    V = []
    W = []
    
    a_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, 1]]
    ]
    
    coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for k in range(7):
        for s in range(rank_sub):
            u_new = [0] * (n * n)
            v_new = [0] * (n * n)
            w_new = [0] * (n * n)
            
            for r in range(2):
                for c in range(2):
                    coeff_a = a_coeffs[k][r][c]
                    coeff_b = b_coeffs[k][r][c]
                    coeff_w = coeffs[k][r][c]
                    
                    if coeff_a != 0:
                        for i in range(n // 2):
                            for j in range(n // 2):
                                idx = (r * (n // 2) + i) * n + (c * (n // 2) + j)
                                u_new[idx] += coeff_a * U_sub[s][i * (n // 2) + j]
                                
                    if coeff_b != 0:
                        for i in range(n // 2):
                            for j in range(n // 2):
                                idx = (r * (n // 2) + i) * n + (c * (n // 2) + j)
                                v_new[idx] += coeff_b * V_sub[s][i * (n // 2) + j]
                                
                    if coeff_w != 0:
                        for i in range(n // 2):
                            for j in range(n // 2):
                                idx = (r * (n // 2) + i) * n + (c * (n // 2) + j)
                                w_new[idx] += coeff_w * W_sub[s][i * (n // 2) + j]
                                
            U.append(u_new)
            V.append(v_new)
            W.append(w_new)
            
    return U, V, W
