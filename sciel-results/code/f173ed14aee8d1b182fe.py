import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices (local): 0:00, 1:01, 2:10, 3:11
    # W indices (local): 0:00, 1:01, 2:10, 3:11 (k*n+i -> k*2+i)
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U contributions (same for all blocks)
    # Col 0 (M1): U[0]=1, U[3]=1
    # Col 1 (M2): U[2]=1, U[3]=1
    # Col 2 (M3): U[0]=1
    # Col 3 (M4): U[3]=1
    # Col 4 (M5): U[0]=1, U[1]=1
    # Col 5 (M6): U[2]=1, U[0]=-1
    # Col 6 (M7): U[1]=1, U[3]=-1
    
    U_defs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    # V contributions (local)
    # Col 0 (M1): V[0]=1, V[3]=1
    # Col 1 (M2): V[0]=1
    # Col 2 (M3): V[1]=1, V[3]=-1
    # Col 3 (M4): V[2]=1, V[0]=-1
    # Col 4 (M5): V[3]=1
    # Col 5 (M6): V[0]=1, V[1]=1
    # Col 6 (M7): V[2]=1, V[3]=1
    
    V_defs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]
    
    # W contributions (local)
    # Local k, i indices: k in {0,1}, i in {0,1}
    # W index = k*2 + i
    # 0: k=0,i=0; 1: k=0,i=1; 2: k=1,i=0; 3: k=1,i=1
    
    # C00 = M1 + M4 - M5 + M7 -> W[0]
    # C01 = M3 + M5 -> W[2]
    # C10 = M2 + M4 -> W[1]
    # C11 = M1 - M2 + M3 + M6 -> W[3]
    
    W_defs = [
        {0: 1, 3: 1},      # M1: C00+=1, C11+=1
        {1: 1, 3: -1},     # M2: C10+=1, C11-=1
        {2: 1, 3: 1},      # M3: C01+=1, C11+=1
        {0: 1, 1: 1},      # M4: C00+=1, C10+=1
        {0: -1, 2: 1},     # M5: C00-=1, C01+=1
        {3: 1},            # M6: C11+=1
        {0: 1}             # M7: C00+=1
    ]
    
    # Fill U
    for r in range(7):
        for idx, val in U_defs[r].items():
            U[idx][r] = val
            
    # Fill V and W for 3 blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            col = block * 7 + r
            # V
            for l_idx, val in V_defs[r].items():
                # l_idx is j*2 + k_loc
                j = l_idx // 2
                k_loc = l_idx % 2
                k_glob = k_offset + k_loc
                v_idx = j * p + k_glob
                V[v_idx][col] = val
                
            # W
            for l_idx, val in W_defs[r].items():
                # l_idx is k_loc*2 + i
                k_loc = l_idx // 2
                i = l_idx % 2
                k_glob = k_offset + k_loc
                w_idx = k_glob * n + i
                W[w_idx][col] = val
                
    if time.time() > deadline:
         # Should not happen, but just in case
         pass

    return {"U": U, "V": V, "W": W}
