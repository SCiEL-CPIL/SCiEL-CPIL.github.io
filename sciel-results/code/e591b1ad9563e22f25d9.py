import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank = 63
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for t in range(7):
        if t == 0:
            u_blocks = {0: 1, 3: 1}
            v_blocks = {0: 1, 3: 1}
            w_coeffs = {0: 1, 3: 1}
        elif t == 1:
            u_blocks = {2: 1, 3: 1}
            v_blocks = {0: 1}
            w_coeffs = {2: 1, 3: -1}
        elif t == 2:
            u_blocks = {0: 1}
            v_blocks = {2: 1, 3: -1}
            w_coeffs = {1: 1, 3: 1}
        elif t == 3:
            u_blocks = {3: 1}
            v_blocks = {1: 1, 0: -1}
            w_coeffs = {0: 1, 2: 1}
        elif t == 4:
            u_blocks = {0: 1, 1: 1}
            v_blocks = {3: 1}
            w_coeffs = {0: -1, 1: 1}
        elif t == 5:
            u_blocks = {2: 1, 0: -1}
            v_blocks = {0: 1, 1: 1}
            w_coeffs = {3: 1}
        elif t == 6:
            u_blocks = {1: 1, 3: -1}
            v_blocks = {2: 1, 3: 1}
            w_coeffs = {0: 1}
            
        for k in range(3):
            for b_idx, coeff in u_blocks.items():
                if b_idx == 0:
                    for i in range(3):
                        U[i * 6 + k, col] += coeff
                elif b_idx == 1:
                    for i in range(3):
                        U[i * 6 + (3+k), col] += coeff
                elif b_idx == 2:
                    for i in range(3):
                        U[(3+i) * 6 + k, col] += coeff
                elif b_idx == 3:
                    for i in range(3):
                        U[(3+i) * 6 + (3+k), col] += coeff
            
            for b_idx, coeff in v_blocks.items():
                if b_idx == 0:
                    for j in range(3):
                        V[k * 6 + j, col] += coeff
                elif b_idx == 1:
                    for j in range(3):
                        V[k * 6 + (3+j), col] += coeff
                elif b_idx == 2:
                    for j in range(3):
                        V[k * 6 + j, col] += coeff
                elif b_idx == 3:
                    for j in range(3):
                        V[k * 6 + (3+j), col] += coeff
            
            for c_idx, coeff in w_coeffs.items():
                if c_idx == 0:
                    for i in range(3):
                        for j in range(3):
                            W[j * 6 + i, col] += coeff
                elif c_idx == 1:
                    for i in range(3):
                        for j in range(3):
                            W[(3+j) * 6 + i, col] += coeff
                elif c_idx == 2:
                    for i in range(3):
                        for j in range(3):
                            W[j * 6 + (3+i), col] += coeff
                elif c_idx == 3:
                    for i in range(3):
                        for j in range(3):
                            W[(3+j) * 6 + (3+i), col] += coeff
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
