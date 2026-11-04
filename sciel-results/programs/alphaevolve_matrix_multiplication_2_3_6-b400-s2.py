import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33: 3 blocks * (7 Strassen + 4 Standard)
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2
    # U, V, W are 2x2 matrices
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 1], [1, -1]]), # W[k,i] -> W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=-1?
        # Check C eqs:
        # C00 (k=0,i=0): M1+M4-M5+M7 -> W[0,0] for M1 is 1
        # C11 (k=1,i=1): M1-M2+M3+M6 -> W[1,1] for M1 is 1
        # C01 (k=1,i=0): M3+M5 -> W[1,0] for M1 is 0
        # C10 (k=0,i=1): M2+M4 -> W[0,1] for M1 is 0
        # So W1 should be [[1, 0], [0, 1]]?
        # Let's re-evaluate W from C equations carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W[r][k, i]
        
        # r=1 (M1): C00(1), C11(1). W[1] = [[1, 0], [0, 1]]
        # r=2 (M2): C10(1), C11(-1). W[2] = [[0, 1], [-1, 0]]? No, W[k,i].
        # C10 is k=0, i=1. So W[2][0,1] = 1.
        # C11 is k=1, i=1. So W[2][1,1] = -1.
        # W[2] = [[0, 1], [0, -1]]
        # r=3 (M3): C01(1), C11(1).
        # C01 is k=1, i=0. W[3][1,0] = 1.
        # C11 is k=1, i=1. W[3][1,1] = 1.
        # W[3] = [[0, 0], [1, 1]]
        # r=4 (M4): C00(1), C10(1).
        # C00 is k=0, i=0. W[4][0,0] = 1.
        # C10 is k=0, i=1. W[4][0,1] = 1.
        # W[4] = [[1, 1], [0, 0]]
        # r=5 (M5): C00(-1), C01(1).
        # C00 is k=0, i=0. W[5][0,0] = -1.
        # C01 is k=1, i=0. W[5][1,0] = 1.
        # W[5] = [[-1, 0], [1, 0]]
        # r=6 (M6): C11(1).
        # C11 is k=1, i=1. W[6][1,1] = 1.
        # W[6] = [[0, 0], [0, 1]]
        # r=7 (M7): C00(1).
        # C00 is k=0, i=0. W[7][0,0] = 1.
        # W[7] = [[1, 0], [0, 0]]
    ]
    
    # Corrected S_W
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Standard factors for 2x1x2
    # A is 2x1 (j=0 local), B is 1x2 (j=0 local), C is 2x2
    # T1: A00 B00 C00
    # T2: A00 B01 C01
    # T3: A10 B00 C10
    # T4: A10 B01 C11
    Std_U = [
        np.array([[1], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [1]])
    ]
    Std_V = [
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 0]]),
        np.array([[0, 1]])
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]]), # C00
        np.array([[0, 0], [1, 0]]), # C01 -> k=1, i=0
        np.array([[0, 1], [0, 0]]), # C10 -> k=0, i=1
        np.array([[0, 0], [0, 1]])  # C11 -> k=1, i=1
    ]
    
    col = 0
    for b in range(3):
        # Strassen part
        for r in range(7):
            # Map U (2x2) to global U (2x2 part of 2x3)
            # Local i, j in 0,1. Global i, j in 0,1.
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            
            # Map V (2x2) to global V (2x2 part of 3x2 block)
            # Local j, k' in 0,1. Global j in 0,1. Global k = 2b + k'.
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            
            # Map W (2x2) to global W (2x2 part of 2x2 block)
            # Local k', i in 0,1. Global k = 2b + k'. Global i in 0,1.
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            
            col += 1
            
        # Standard part
        for r in range(4):
            # Map U (2x1) to global U (2x1 part at j=2)
            for i in range(2):
                # Local j is 0, Global j is 2
                U[i * m + 2, col] = Std_U[r][i, 0]
            
            # Map V (1x2) to global V (1x2 part at j=2)
            for kp in range(2):
                k_global = 2 * b + kp
                # Local j is 0, Global j is 2
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            
            # Map W (2x2) to global W
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
