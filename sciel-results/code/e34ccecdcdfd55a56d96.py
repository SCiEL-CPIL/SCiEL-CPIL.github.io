import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def _validate(U, V, W, tol=1e-9):
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_vectors():
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

def construct():
    t0 = time.time()
    n, m, p = N, M, P
    R = 85
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    c = 0
    
    s2 = _strassen_2x2_vectors()
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    
    # Strassen 4x4 calls
    # 1. (A00+A11)(B00+B11) -> C00+C11
    # 2. (A10+A11)B00 -> C10-C11
    # 3. A00(B01-B11) -> C01+C11
    # 4. A11(B10-B00) -> C00+C10
    # 5. (A00+A01)B11 -> -C00+C01
    # 6. (A10-A00)(B00+B01) -> C11
    # 7. (A01-A11)(B10+B11) -> C00
    
    # We define the block offsets for each call.
    # A blocks: 00:(0,0), 01:(0,2), 10:(2,0), 11:(2,2)
    # B blocks: 00:(0,0), 01:(0,2), 10:(2,0), 11:(2,2)
    # C blocks: 00:(0,0), 01:(0,2), 10:(2,0), 11:(2,2)
    
    # For each call, we sum the 7 Strassen 2x2 terms.
    # Each term has u, v, w vectors of length 4.
    # We map them to length 16 vectors at the block offsets.
    
    # Helper to map a 2x2 vector to a 4x4 vector at offset
    def map_2x2_to_4x4(vec, row_off, col_off):
        out = [0.0] * 16
        for r in range(2):
            for c in range(2):
                out[(row_off + r) * 4 + (col_off + c)] = vec[r * 2 + c]
        return out

    # Define the 7 calls
    # Each call is a list of (u_blocks, v_blocks, w_blocks)
    # u_blocks is a list of (row_off, col_off, sign)
    
    calls = [
        # 1
        ([ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ]),
        # 2
        ([ (2,0,1), (2,2,1) ], [ (0,0,1) ], [ (2,0,1), (2,2,-1) ]),
        # 3
        ([ (0,0,1) ], [ (0,2,1), (2,2,-1) ], [ (0,2,1), (2,2,1) ]),
        # 4
        ([ (2,2,1) ], [ (2,0,1), (0,0,-1) ], [ (0,0,1), (2,0,1) ]),
        # 5
        ([ (0,0,1), (0,2,1) ], [ (2,2,1) ], [ (0,0,-1), (0,2,1) ]),
        # 6
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (0,2,1) ], [ (2,2,1) ]),
        # 7
        ([ (0,2,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ], [ (0,0,1) ])
    ]
    
    for u_blocks, v_blocks, w_blocks in calls:
        for u_vec, v_vec, w_vec in s2:
            # Construct full 4x4 vectors
            u_full = [0.0] * 16
            v_full = [0.0] * 16
            w_full = [0.0] * 16
            
            for row_off, col_off, sign in u_blocks:
                mapped = map_2x2_to_4x4(u_vec, row_off, col_off)
                for i in range(16):
                    u_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in v_blocks:
                mapped = map_2x2_to_4x4(v_vec, row_off, col_off)
                for i in range(16):
                    v_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in w_blocks:
                mapped = map_2x2_to_4x4(w_vec, row_off, col_off)
                for i in range(16):
                    w_full[i] += sign * mapped[i]
            
            # Add to global U, V, W
            # A11 is A[0:4, 0:4]. Global U index: i*5+j
            # B111 is B[0:4, 0:4]. Global V index: j*5+k
            # C11 is C[0:4, 0:4]. Global W index: k*4+i
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * 5 + j
                    u_val = u_full[i * 4 + j]
                    if u_val != 0:
                        U[u_idx, c] += u_val
                        
            for j in range(4):
                for k in range(4):
                    v_idx = j * 5 + k
                    v_val = v_full[j * 4 + k]
                    if v_val != 0:
                        V[v_idx, c] += v_val
                        
            for k in range(4):
                for i in range(4):
                    w_idx = k * 4 + i
                    w_val = w_full[k * 4 + i]
                    if w_val != 0:
                        W[w_idx, c] += w_val
                        
            c += 1
            
    # Part 2: A11 B112 (4x4x1) Rank 16
    # A11: A[0:4, 0:4]. B112: B[0:4, 4]. C12: C[0:4, 4].
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0
            c += 1
            
    # Part 3: A12 B21 (4x1x5) Rank 20
    # A12: A[0:4, 4]. B21: B[4, 0:5]. C2: C[0:4, 0:5].
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, c] = 1.0
            V[4 * 5 + k, c] = 1.0
            W[k * 4 + i, c] = 1.0
            c += 1
            
    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
