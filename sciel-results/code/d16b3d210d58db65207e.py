import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Constructs a rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Decomposition Strategy:
    Split n=3 into n1=1 and n2=2.
    M(3,4,8) = M(1,4,8) + M(2,4,8)
    
    1. M(1,4,8) uses schoolbook multiplication: Rank = 1*4*8 = 32.
    2. M(2,4,8) is decomposed using Strassen's algorithm on 2x2 blocks.
       Split m=4 into two blocks of 2: m1=2, m2=2.
       Split p=8 into four blocks of 2: p1=2, p2=2, p3=2, p4=2.
       This creates 8 independent 2x2x2 sub-problems.
       Each 2x2x2 problem uses Strassen (Rank 7).
       Total Rank for M(2,4,8) = 8 * 7 = 56.
       
    Total Rank = 32 + 56 = 88.
    """
    n, m, p = DIMENSIONS
    
    # Dimensions
    n1, n2 = 1, 2
    m1, m2 = 2, 2
    p_blocks = [2, 2, 2, 2]
    
    rank = 88
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part 1: M(1,4,8)
    # Indices: i=0, j=0..3, k=0..7
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: M(2,4,8)
    # Indices: i=1,2 (local 0,1), j=0..3, k=0..7
    
    # Strassen factors for 2x2x2
    # Local indices for 2x2 block:
    # A: [[a00, a01], [a10, a11]] -> flat [a00, a01, a10, a11]
    # B: [[b00, b01], [b10, b11]] -> flat [b00, b01, b10, b11]
    # C: [[c00, c01], [c10, c11]] -> flat [c00, c01, c10, c11]
    
    # Strassen matrices (7x4)
    S_U = np.array([
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ])
    
    S_V = np.array([
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ])
    
    S_W = np.array([
        [1, 0, 0, 1], # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1], # M3: C01+=1, C11+=1
        [1, 0, 1, 0], # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1], # M6: C11+=1
        [1, 0, 0, 0]  # M7: C00+=1
    ])
    
    # Iterate over j_blocks (2) and k_blocks (4)
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # Update U
                # Local A indices l=0..3 map to i_local=0,1 and j_local=0,1
                # l=0 -> (0,0), l=1 -> (0,1), l=2 -> (1,0), l=3 -> (1,1)
                for l in range(4):
                    if S_U[t, l] != 0:
                        i_local = l // 2
                        j_local = l % 2
                        i_global = n1 + i_local # 1 + i_local
                        j_global = j_start + j_local
                        U[i_global * m + j_global][col] += S_U[t, l]
                        
                # Update V
                # Local B indices l=0..3 map to j_local=0,1 and k_local=0,1
                for l in range(4):
                    if S_V[t, l] != 0:
                        j_local = l // 2
                        k_local = l % 2
                        j_global = j_start + j_local
                        k_global = k_start + k_local
                        V[j_global * p + k_global][col] += S_V[t, l]
                        
                # Update W
                # Local C indices l=0..3 map to i_local=0,1 and k_local=0,1
                for l in range(4):
                    if S_W[t, l] != 0:
                        i_local = l // 2
                        k_local = l % 2
                        i_global = n1 + i_local # 1 + i_local
                        k_global = k_start + k_local
                        W[k_global * n + i_global][col] += S_W[t, l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
