import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: (16, R), V: (24, R), W: (24, R)
    
    # Strassen coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # alpha[s][r] is coeff for A_r in X_s
    # beta[s][c] is coeff for B_c in Y_s
    # recon[s] is list of (a, b, gamma)
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    # Block indices mapping
    # A_r: block (ar, br). ar = r//2, br = r%2
    # B_c: block (ac, bc). ac = c//2, bc = c%2
    # C_ab: block (a, b)
    
    R = 84
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        for u in range(2):
            for v in range(2):
                for w in range(3):
                    # This corresponds to the term X_s[u,v] * Y_s[v,w]
                    # U part
                    for r in range(4):
                        if alpha[s][r] != 0:
                            # A_r[u,v]
                            ar, br = r // 2, r % 2
                            # Global index in A (4x4)
                            # Row: ar*2 + u
                            # Col: br*2 + v
                            u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                            U[u_idx, col] += alpha[s][r]
                            
                    # V part
                    for c in range(4):
                        if beta[s][c] != 0:
                            # B_c[v,w]
                            ac, bc = c // 2, c % 2
                            # Global index in B (4x6)
                            # Row: ac*2 + v
                            # Col: bc*3 + w
                            v_idx = (ac * 2 + v) * 6 + (bc * 3 + w)
                            V[v_idx, col] += beta[s][c]
                            
                    # W part
                    for a, b, gamma in recon[s]:
                        # C_ab[u,w]
                        # Global index in C (4x6)
                        # Row: a*2 + u
                        # Col: b*3 + w
                        # W index: Col * 4 + Row
                        w_idx = (b * 3 + w) * 4 + (a * 2 + u)
                        W[w_idx, col] += gamma
                        
                    col += 1
                    
    # Convert to lists
    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
