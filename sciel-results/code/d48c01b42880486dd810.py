import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, num_trials=5):
    """Verify exact integer decomposition using small integer test matrices."""
    rank = len(U[0])
    np.random.seed(42)
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_true = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            x = np.dot(A.flatten(), [U[i*m+j][r] for i in range(n) for j in range(m)])
            y = np.dot(B.flatten(), [V[j*p+k][r] for j in range(m) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += x * y * W[k*n+i][r]
        if not np.array_equal(C_true, C_decomp):
            return False
    return True

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Macro factors for 2x2 blocks
    # U_macro, V_macro, W_macro are lists of 7 vectors of length 4
    # Indices 0,1,2,3 correspond to blocks (0,0), (0,1), (1,0), (1,1)
    
    # Micro factors for 2x2x2 Strassen (Rank 7)
    # u_micro, v_micro, w_micro are lists of 7 vectors of length 4
    # Indices 0,1,2,3 correspond to elements (0,0), (0,1), (1,0), (1,1)
    
    # Micro Strassen Factors
    # M1: (A00+A11)(B00+B11) -> W: C00+=M1, C11+=M1
    # M2: (A10+A11)B00 -> W: C01+=M2
    # M3: A00(B01-B11) -> W: C01+=M3, C11-=M3
    # M4: A11(B10-B00) -> W: C00+=M4, C11+=M4
    # M5: (A00+A01)B11 -> W: C00-=M5, C11+=M5
    # M6: (A10-A00)(B00+B01) -> W: C10+=M6
    # M7: (A01-A11)(B10+B11) -> W: C00-=M7, C10+=M7
    
    u_micro = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    v_micro = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_micro = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C01
        [0, 1, 0, -1],# M3: C01, C11(-)
        [1, 0, 0, 1], # M4: C00, C11
        [-1, 0, 0, 1],# M5: C00(-), C11
        [0, 0, 1, 0], # M6: C10
        [-1, 0, 1, 0] # M7: C00(-), C10
    ]
    
    # Macro Strassen Factors (same structure as micro, but applied to 2x2 blocks)
    u_macro = u_micro
    v_macro = v_micro
    w_macro = w_micro
    
    col = 0
    for k in range(7):
        for l in range(7):
            # U term: u_macro[k] (4 blocks) x u_micro[l] (4 elems) -> 16 entries
            # Map block b (0..3) to block row/col in 2x2 grid
            # b=0: (0,0), b=1: (0,1), b=2: (1,0), b=3: (1,1)
            # Within block, offset o (0..3) corresponds to elem (0,0), (0,1), (1,0), (1,1)
            # Global A index: i = block_row * 2 + elem_row, j = block_col * 2 + elem_col
            # U index: i * m + j = i * 5 + j
            
            for b in range(4):
                if u_macro[k][b] == 0: continue
                # Determine block row/col
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if u_micro[l][o] == 0: continue
                    # Determine elem row/col
                    er = o // 2
                    ec = o % 2
                    # Global i, j
                    gi = br * 2 + er
                    gj = bc * 2 + ec
                    # U index
                    ui = gi * 5 + gj
                    U[ui][col] += u_macro[k][b] * u_micro[l][o]
            
            # V term: v_macro[k] x v_micro[l]
            for b in range(4):
                if v_macro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if v_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    # Global j, k (B is 5x5, but we use 4x4 part)
                    gj = br * 2 + er
                    gk = bc * 2 + ec
                    # V index: j * p + k = j * 5 + k
                    vi = gj * 5 + gk
                    V[vi][col] += v_macro[k][b] * v_micro[l][o]
            
            # W term: w_macro[k] x w_micro[l]
            for b in range(4):
                if w_macro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if w_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    # Global i, k (C is 4x5, but we use 4x4 part)
                    gi = br * 2 + er
                    gk = bc * 2 + ec
                    # W index: k * n + i = k * 4 + i
                    wi = gk * 4 + gi
                    W[wi][col] += w_macro[k][b] * w_micro[l][o]
            
            col += 1
    
    # Part 2: 4x4x1 (Rank 16)
    # A(4x4) * B(4x1) -> C(4x1)
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k=4
    # C indices: i in 0..3, k=4
    for i in range(4):
        for j in range(4):
            ui = i * 5 + j
            vi = j * 5 + 4
            wi = 4 * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    # Part 3: 4x1x5 (Rank 20)
    # A(4x1) * B(1x5) -> C(4x5)
    # A indices: i in 0..3, j=4
    # B indices: j=4, k in 0..4
    # C indices: i in 0..3, k in 0..4
    for i in range(4):
        for k in range(5):
            ui = i * 5 + 4
            vi = 4 * 5 + k
            wi = k * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to Kronecker if verification fails (should not happen)
        rank_k = n * m * p
        Uk = [[0]*rank_k for _ in range(n*m)]
        Vk = [[0]*rank_k for _ in range(m*p)]
        Wk = [[0]*rank_k for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    Uk[i*m+j][r] = 1
                    Vk[j*p+k][r] = 1
                    Wk[k*n+i][r] = 1
        return {"U": Uk, "V": Vk, "W": Wk}
