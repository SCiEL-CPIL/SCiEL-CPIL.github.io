import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 77
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen coefficients for 2x2 * 2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            # U: combine alpha[s] (outer) with alpha[si] (inner)
            for r in range(4):
                if alpha[s][r] != 0:
                    ar, br = r // 2, r % 2
                    for a in range(4):
                        if alpha[si][a] != 0:
                            x_row, x_col = a // 2, a % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[s][r] * alpha[si][a]
            
            # V: combine beta[s] (outer) with beta[si] (inner) for the 2x2 part (cols 0,1)
            for c in range(4):
                if beta[s][c] != 0:
                    ac, bc = c // 2, c % 2
                    for b in range(4):
                        if beta[si][b] != 0:
                            y_row, y_inner_col = b // 2, b % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 2 + y_inner_col)
                            V[v_idx, col] += beta[s][c] * beta[si][b]
            
            # W: combine recon[s] (outer) with recon[si] (inner) for the 2x2 part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        # Now the 2x1 part (column 2 of the 2x3 block)
        for u in range(2):
            for v in range(2):
                # U: alpha[s] only (no inner si, since 2x1 schoolbook is just 4 terms)
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V: beta[s] for the 2x1 part (column index 2)
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 2 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W: recon[s] for the 2x1 part (column index 2)
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            result_row = []
            for x in row:
                rounded = round(x)
                if abs(x - rounded) < 1e-9:
                    result_row.append(int(rounded))
                else:
                    result_row.append(x)
            result.append(result_row)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
