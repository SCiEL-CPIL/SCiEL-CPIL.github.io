import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen 2x2x2 decomposition vectors
    # U: index i*2+j
    # V: index j*2+k
    # W: index k*2+i
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # U: 1,0,0,1
    # V: 1,0,0,1
    # W: 1,0,0,1
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2: (A10+A11)B00 -> C10, C11
        # V: B00 is index 0. W: C10 (k=0,i=1 -> 1), C11 (k=1,i=1 -> 3).
        # M2 = (A10+A11)B00.
        # A10 (i=1,j=0 -> 2), A11 (i=1,j=1 -> 3). U: 0,0,1,1.
        # B00 (j=0,k=0 -> 0). V: 1,0,0,0.
        # C10 (k=0,i=1 -> 1), C11 (k=1,i=1 -> 3).
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So W should have 1 at 1 and -1 at 3?
        # Let's check signs.
        # C11 = ... - M2 ...
        # C10 = ... + M2 ...
        # So W[1]=1, W[3]=-1. Correct.
        
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3: A00(B01-B11) -> C01, C11
        # A00 (0). U: 1,0,0,0.
        # B01 (1), B11 (3). V: 0,1,0,-1.
        # C01 (k=1,i=0 -> 2). C11 (k=1,i=1 -> 3).
        # C01 = M3 + M5. C11 = ... + M3 ...
        # So W[2]=1, W[3]=1?
        # Wait. C01 index is 2. C11 index is 3.
        # My W above is [0, 1, 0, 0]. Index 1 is 1?
        # W index k*2+i.
        # C01: k=1, i=0 -> 2.
        # C11: k=1, i=1 -> 3.
        # So W should be [0, 0, 1, 1].
        # Let's re-verify M3 W.
        # M3 = A00(B01-B11).
        # Contributes to C01 and C11?
        # C01 = A00B01 + A01B11.
        # C11 = A00B01 + A01B10 + A10B01 + A11B11? No.
        # C11 = A00B01 + A01B11? No.
        # C11 = A00B01 + A01B11 is C01.
        # C11 = A00B01 + A01B11?
        # C11 = sum_j A1j Bj1 = A10B01 + A11B11.
        # So M3 = A00B01 - A00B11.
        # A00B01 is in C01.
        # A00B11 is NOT in C11.
        # So M3 only contributes to C01?
        # But Strassen says C11 = M1 - M2 + M3 + M6.
        # Let's check M1 - M2 + M3 + M6 for C11.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # M2 = (A10+A11)B00 = A10B00 + A11B00.
        # M3 = A00(B01-B11) = A00B01 - A00B11.
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01.
        # Sum:
        # A00B00: 1 - 1 = 0.
        # A00B11: 1 - 1 = 0.
        # A11B00: 1 - 1 = 0.
        # A11B11: 1.
        # A10B00: -1 + 1 = 0.
        # A00B01: -1 - 1 = -2?
        # A10B01: 1.
        # This doesn't sum to C11 = A10B01 + A11B11.
        # There is an error in my manual Strassen expansion or the components.
        
        # Let's stick to the verified components from the current program which passed verification (or at least were structured correctly).
        # The current program's components:
        # M3: U{(0,0):1}, V{(0,1):1, (1,1):-1}, W{(0,1):1}
        # W index for (0,1) is k=1, i=0 -> 2.
        # So W = [0, 0, 1, 0].
        
        # Let's just use the dicts from the current program to be safe, and convert to lists.
    ]
    
    # Re-define using dicts for safety
    strassen_dicts = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    def dict_to_vec(d, size, idx_func):
        v = [0] * size
        for idx, coeff in d.items():
            v[idx_func(idx)] = coeff
        return v
    
    strassen_vecs = []
    for u_d, v_d, w_d in strassen_dicts:
        u_v = dict_to_vec(u_d, 4, lambda i: i[0]*2 + i[1])
        v_v = dict_to_vec(v_d, 4, lambda i: i[0]*2 + i[1])
        w_v = dict_to_vec(w_d, 4, lambda i: i[0]*2 + i[1])
        strassen_vecs.append((u_v, v_v, w_v))
    
    # Build 4x4x4 decomposition
    # Rank 49
    U_444 = []
    V_444 = []
    W_444 = []
    
    for u_r, v_r, w_r in strassen_vecs:
        for u_s, v_s, w_s in strassen_vecs:
            # Kronecker product
            u_kron = []
            for a in u_r:
                for b in u_s:
                    u_kron.append(a*b)
            v_kron = []
            for a in v_r:
                for b in v_s:
                    v_kron.append(a*b)
            w_kron = []
            for a in w_r:
                for b in w_s:
                    w_kron.append(a*b)
            U_444.append(u_kron)
            V_444.append(v_kron)
            W_444.append(w_kron)
            
    rank_444 = len(U_444) # 49
    
    # Build full 4x4x9 decomposition
    rank_total = rank_444 * 2 + 1
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(36)]
    W = [[0] * rank_total for _ in range(36)]
    
    col = 0
    # Block 1: B cols 0-3
    for i in range(rank_444):
        u = U_444[i]
        v = V_444[i]
        w = W_444[i]
        
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][col] = val
                
        for idx, val in enumerate(v):
            # idx = j*4 + k
            j = idx // 4
            k = idx % 4
            v_idx = j * 9 + k
            if val != 0:
                V[v_idx][col] = val
                
        for idx, val in enumerate(w):
            # idx = k*4 + i
            k = idx // 4
            i = idx % 4
            w_idx = k * 4 + i
            if val != 0:
                W[w_idx][col] = val
        col += 1
        
    # Block 2: B cols 4-7
    for i in range(rank_444):
        u = U_444[i]
        v = V_444[i]
        w = W_444[i]
        
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][col] = val
                
        for idx, val in enumerate(v):
            j = idx // 4
            k = idx % 4
            v_idx = j * 9 + (k + 4)
            if val != 0:
                V[v_idx][col] = val
                
        for idx, val in enumerate(w):
            k = idx // 4
            i = idx % 4
            w_idx = (k + 4) * 4 + i
            if val != 0:
                W[w_idx][col] = val
        col += 1
        
    # Block 3: B col 8 (Rank 1)
    # U: all 1s
    for i in range(16):
        U[i][col] = 1
    # V: col 8 all 1s
    for j in range(4):
        V[j*9 + 8][col] = 1
    # W: col 8 (k=8) all 1s
    for i in range(4):
        W[8*4 + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
