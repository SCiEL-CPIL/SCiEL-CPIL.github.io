import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Rank-7 decomposition of 2x2 matrix multiplication
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11+A22, A21+A22, A11, A22, A11+A12, A21-A11, A12-A22
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11+B22, B11, B12-B22, B21-B11, B22, B11+B12, B21+B22
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11+C22, C21, C12, C11+C21, C11-C12, C22, C11+C12
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Strassen steps
    # k=1..7
    # Xk, Yk definitions
    # We define S_A[k] and S_B[k] as 4x16 matrices
    # Row i,j of S_A[k] gives coefficients for A[0..3, 0..3] to form Xk[i,j]
    
    S_A = [None] * 7
    S_B = [None] * 7
    beta = [None] * 7 # beta[k] is 4x4 (block coeffs) or just a dict
    
    # Helper to create S matrix
    def make_S(blocks):
        # blocks is list of (block_idx, coeff)
        # block_idx 0:11, 1:12, 2:21, 3:22
        S = np.zeros((4, 16), dtype=int)
        for b_idx, coeff in blocks:
            # Map block to rows/cols in 4x4
            if b_idx == 0: # 11
                r0, c0 = 0, 0
            elif b_idx == 1: # 12
                r0, c0 = 0, 2
            elif b_idx == 2: # 21
                r0, c0 = 2, 0
            else: # 22
                r0, c0 = 2, 2
            
            for i in range(2):
                for j in range(2):
                    a_idx = (r0 + i) * 4 + (c0 + j)
                    x_idx = i * 2 + j
                    S[x_idx, a_idx] += coeff
        return S
    
    # k=1: X=A11+A22, Y=B11+B22
    S_A[0] = make_S([(0, 1), (3, 1)])
    S_B[0] = make_S([(0, 1), (3, 1)])
    
    # k=2: X=A21+A22, Y=B11
    S_A[1] = make_S([(2, 1), (3, 1)])
    S_B[1] = make_S([(0, 1)])
    
    # k=3: X=A11, Y=B12-B22
    S_A[2] = make_S([(0, 1)])
    S_B[2] = make_S([(1, 1), (3, -1)])
    
    # k=4: X=A22, Y=B21-B11
    S_A[3] = make_S([(3, 1)])
    S_B[3] = make_S([(2, 1), (0, -1)])
    
    # k=5: X=A11+A12, Y=B22
    S_A[4] = make_S([(0, 1), (1, 1)])
    S_B[4] = make_S([(3, 1)])
    
    # k=6: X=A21-A11, Y=B11+B12
    S_A[5] = make_S([(2, 1), (0, -1)])
    S_B[5] = make_S([(0, 1), (1, 1)])
    
    # k=7: X=A12-A22, Y=B21+B22
    S_A[6] = make_S([(1, 1), (3, -1)])
    S_B[6] = make_S([(2, 1), (3, 1)])
    
    # Beta coefficients for C blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # beta[k][block]
    # block 0:11, 1:12, 2:21, 3:22
    beta_coeffs = [
        [1, 0, 0, 0], # k=1: C11=1, C22=1
        [0, 0, 1, 0], # k=2: C21=1, C22=-1
        [0, 1, 0, 0], # k=3: C12=1, C22=1
        [1, 0, 1, 0], # k=4: C11=1, C21=1
        [-1, 1, 0, 0], # k=5: C11=-1, C12=1
        [0, 0, 0, 1], # k=6: C22=1
        [1, 1, 0, 0]  # k=7: C11=1, C12=1
    ]
    # Wait, k=1 affects C11 and C22.
    # My beta_coeffs list above is indexed by k, and has 4 entries for blocks 11,12,21,22.
    # Let's re-verify.
    # k=1: C11+=1, C22+=1. So [1, 0, 0, 1]
    # k=2: C21+=1, C22+=-1. So [0, 0, 1, -1]
    # k=3: C12+=1, C22+=1. So [0, 1, 0, 1]
    # k=4: C11+=1, C21+=1. So [1, 0, 1, 0]
    # k=5: C11+=-1, C12+=1. So [-1, 1, 0, 0]
    # k=6: C22+=1. So [0, 0, 0, 1]
    # k=7: C11+=1, C12+=1. So [1, 1, 0, 0]
    
    beta_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ]
    
    for k in range(7):
        for r in range(7):
            R = k * 7 + r
            # U
            # U[a, R] = sum_x U2[x, r] * S_A[k][x, a]
            # S_A[k] is 4x16. U2[:, r] is 4x1.
            # Result is 16x1.
            U[:, R] = S_A[k].T @ U2[:, r]
            V[:, R] = S_B[k].T @ V2[:, r]
            
            # W
            # W[c, R] = beta[k][block] * W2[y, r]
            # c is index in 4x4 C.
            # y is index in 2x2 Zk.
            # block is determined by c.
            # c = row * 4 + col.
            # row = 2*u + i, col = 2*v + j.
            # y = i * 2 + j.
            for c in range(16):
                row = c // 4
                col = c % 4
                u = row // 2
                v = col // 2
                i = row % 2
                j = col % 2
                y = i * 2 + j
                block_idx = u * 2 + v # 0:11, 1:12, 2:21, 3:22
                W[c, R] = beta_coeffs[k][block_idx] * W2[y, r]
                
    return {"U": U, "V": V, "W": W}
