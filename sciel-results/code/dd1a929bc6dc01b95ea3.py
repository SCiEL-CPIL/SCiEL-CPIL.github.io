import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

def _build_strassen_4x4():
    """Build Strassen factors for 4x4x4 matrix multiplication."""
    # Strassen for 2x2x2
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
    ])
    V2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
    ])
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
    ])
    
    # Actually, let me use the standard Strassen construction properly
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-c), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # U factors for A (row-major: a=0, b=1, c=2, d=3)
    U2 = np.zeros((4, 7))
    U2[0, 0] = 1  # a in M1
    U2[3, 0] = 1  # d in M1
    U2[2, 1] = 1  # c in M2
    U2[3, 1] = 1  # d in M2
    U2[0, 2] = 1  # a in M3
    U2[3, 3] = 1  # d in M4
    U2[0, 4] = 1  # a in M5
    U2[1, 4] = 1  # b in M5
    U2[2, 5] = 1  # c in M6
    U2[0, 5] = -1 # -a in M6
    U2[1, 6] = 1  # b in M7
    U2[3, 6] = -1 # -d in M7
    
    # V factors for B (row-major: e=0, f=1, g=2, h=3)
    V2 = np.zeros((4, 7))
    V2[0, 0] = 1  # e in M1
    V2[3, 0] = 1  # h in M1
    V2[0, 1] = 1  # e in M2
    V2[1, 2] = 1  # f in M3
    V2[3, 2] = -1 # -h in M3
    V2[2, 3] = 1  # g in M4
    V2[0, 3] = -1 # -c... wait, this is for B
    # M4 = d(g-c), so V for M4: g=1, and we need -c but c is in A
    # Actually V is only for B, so M4: g=1
    V2[2, 3] = 1  # g in M4
    V2[3, 4] = 1  # h in M5
    V2[0, 5] = 1  # e in M6
    V2[1, 5] = 1  # f in M6
    V2[2, 6] = 1  # g in M7
    V2[3, 6] = 1  # h in M7
    
    # W factors for C (row-major: c11=0, c12=1, c21=2, c22=3)
    W2 = np.zeros((4, 7))
    W2[0, 0] = 1  # c11 += M1
    W2[2, 0] = 0  # 
    W2[3, 0] = 0  #
    W2[0, 1] = 0
    W2[2, 1] = 1  # c21 += M2
    W2[1, 2] = 1  # c12 += M3
    W2[0, 3] = 1  # c11 += M4
    W2[2, 3] = 1  # c21 += M4
    W2[0, 4] = -1 # c11 -= M5
    W2[1, 4] = 1  # c12 += M5
    W2[3, 5] = 1  # c22 += M6
    W2[0, 6] = 1  # c11 += M7
    
    return U2, V2, W2

def _strassen_4x4():
    """Build 4x4x4 Strassen factors using 2x2 Strassen."""
    U2, V2, W2 = _build_strassen_4x4()
    n = 4
    n_half = 2
    rank2 = 7
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    A_blocks = [
        {0: 1, 3: 1},  # M1: A11+A22
        {2: 1, 3: 1},  # M2: A21+A22
        {0: 1},        # M3: A11
        {3: 1},        # M4: A22
        {0: 1, 1: 1},  # M5: A11+A12
        {2: 1, 0: -1}, # M6: A21-A11
        {1: 1, 3: -1}, # M7: A12-A22
    ]
    B_blocks = [
        {0: 1, 3: 1},  # M1: B11+B22
        {0: 1},        # M2: B11
        {1: 1, 3: -1}, # M3: B12-B22
        {2: 1, 0: -1}, # M4: B21-B11
        {3: 1},        # M5: B22
        {0: 1, 1: 1},  # M6: B11+B12
        {2: 1, 3: 1},  # M7: B21+B22
    ]
    C_blocks = [
        {0: 1, 3: 1},  # M1: C11+=M1, C22+=M1... wait
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        {0: 1, 2: -1, 1: 0, 3: 0},  # M1: C11+=1, C21-=1? No...
    ]
    
    # Let me redo C coefficients properly
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    C_blocks = [
        {0: 1, 3: 1},  # M1: C11+=1, C22+=1
        {2: 1, 3: -1}, # M2: C21+=1, C22-=1
        {1: 1, 3: 1},  # M3: C12+=1, C22+=1
        {0: 1, 2: 1},  # M4: C11+=1, C21+=1
        {0: -1, 1: 1}, # M5: C11-=1, C12+=1
        {3: 1},        # M6: C22+=1
        {0: 1},        # M7: C11+=1
    ]
    
    for f in range(7):
        for l in range(rank2):
            col = f * rank2 + l
            for b_idx, coeff in A_blocks[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        U4[global_idx, col] += coeff * U2[local_idx, l]
            for b_idx, coeff in B_blocks[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        V4[global_idx, col] += coeff * V2[local_idx, l]
            for b_idx, coeff in C_blocks[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for k in range(n_half):
                    for i in range(n_half):
                        local_idx = k * n_half + i
                        global_idx = (r_start + k) * n + (c_start + i)
                        W4[global_idx, col] += coeff * W2[local_idx, l]
    
    return U4, V4, W4

def _validate(U, V, W, n, m, p):
    """Validate tensor decomposition."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a] * V[b] * W[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_4x8_strassen():
    """Construct 4x4x8 using two 4x4 Strassen blocks."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _strassen_4x4()
    rank4 = U4.shape[1]
    
    U = np.zeros((16, 2 * rank4))
    V = np.zeros((32, 2 * rank4))
    W = np.zeros((32, 2 * rank4))
    
    # First block: k=0..3
    U[:, 0:rank4] = U4
    V[0:16, 0:rank4] = V4
    W[0:16, 0:rank4] = W4
    
    # Second block: k=4..7
    U[:, rank4:2*rank4] = U4
    for r in range(16):
        j = r // 4
        k = r % 4
        V[j * 4 + (k + 4), rank4:2*rank4] = V4[r, :]
    for r in range(16):
        k = r // 4
        i = r % 4
        W[(k + 4) * 4 + i, rank4:2*rank4] = W4[r, :]
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_random_search(U_init, V_init, W_init, n, m, p, max_iter=200, deadline=None):
    """Try to reduce rank by random perturbations and rank reduction."""
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        
        # Try to reduce rank by 1
        new_rank = rank - 1
        if new_rank <= 0:
            break
        
        # Randomly choose a column to eliminate
        col_to_remove = random.randint(0, rank - 1)
        
        # Create new matrices without that column
        cols_keep = [c for c in range(rank) if c != col_to_remove]
        U_new = U[:, cols_keep].copy()
        V_new = V[:, cols_keep].copy()
        W_new = W[:, cols_keep].copy()
        
        # Try to add a new random column to compensate
        # The error is the tensor difference
        # For simplicity, try random small perturbations
        
        # Check if the reduced rank still works (it won't in general)
        # Instead, try to find a linear combination
        pass
    
    return best_rank, best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Start with Strassen-based 4x4x8 (rank 98)
    U, V, W = _construct_4x8_strassen()
    
    if _validate(U, V, W, n, m, p):
        current_rank = len(U[0])
        
        # Try to improve
        # For now, return the Strassen result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
