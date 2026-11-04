import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS
    
    # 1. Build M_{2,2,2} (Strassen, rank 7)
    S222_U = [
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1]
    ]
    S222_V = [
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1]
    ]
    S222_W = [
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 1]
    ]
    
    def build_223():
        # M_{2,2,3} = M_{2,2,2} \oplus M_{2,2,1}
        U = []
        V = []
        W = []
        for r in range(7):
            u = [0,0,0,0]
            for i in range(2):
                for j in range(2):
                    u[i*2+j] = S222_U[i][j][r]
            v = [0,0,0,0]
            for j in range(2):
                for k in range(2):
                    v[j*2+k] = S222_V[j][k][r]
            w = [0,0,0,0]
            for k in range(2):
                for i in range(2):
                    w[k*2+i] = S222_W[k][i][r]
            U.append(u); V.append(v); W.append(w)
        for r in range(4):
            # M_{2,2,1}: k=1
            u = [0,0,1,1]
            v = [0,0,1,1]
            w = [0,0,1,1]
            U.append(u); V.append(v); W.append(w)
        return U, V, W
    
    # 2. Build M_{3,3,3} (Strassen + 1, rank 23)
    # We build M_{3,3,3} by partitioning I into 2+1.
    # T = M_{2,2,3} \oplus M_{1,1,3} \oplus M_{2,1,3} \oplus M_{2,1,3}?
    # Let's use the known rank 23 construction.
    # Actually, we can construct it using M_{2,2,2} (7) and 16 extra terms.
    # Let's use a known explicit list if possible, or fall back to schoolbook if not found in time.
    # Given the complexity, we will use a reliable block construction:
    # M_{3,3,3} = M_{2,2,3} \oplus M_{1,1,3} \oplus M_{2,1,3} \oplus M_{2,1,3}
    # Rank: 11 + 3 + 2 + 2 = 18?
    # Let's check dimensions.
    # M_{3,3,3}: I=3, J=3, K=3.
    # Partition I into 2+1, J into 2+1, K into 3.
    # Terms:
    # 1. I=0,1 J=0,1 K=0,1,2 -> M_{2,2,3} (Rank 11)
    # 2. I=2   J=0,1 K=0,1,2 -> M_{1,2,3} (Rank 6, schoolbook)
    # 3. I=0,1 J=2   K=0,1,2 -> M_{2,1,3} (Rank 6, schoolbook)
    # 4. I=2   J=2   K=0,1,2 -> M_{1,1,3} (Rank 3, schoolbook)
    # Total Rank: 11 + 6 + 6 + 3 = 26.
    # This is better than 27.
    
    def build_333():
        U, V, W = build_223() # 11 terms
        # Add M_{1,2,3} (I=2, J=0,1, K=0,1,2)
        for j in range(2):
            for k in range(3):
                u = [0]*9; u[8] = 1
                v = [0]*9; v[j*3+k] = 1
                w = [0]*9; w[k*3+8] = 1
                U.append(u); V.append(v); W.append(w)
        # Add M_{2,1,3} (I=0,1, J=2, K=0,1,2)
        for i in range(2):
            for k in range(3):
                u = [0]*9; u[i*3+8] = 1
                v = [0]*9; v[8] = 1
                w = [0]*9; w[k*3+i*3] = 1
                U.append(u); V.append(v); W.append(w)
        # Add M_{1,1,3} (I=2, J=2, K=0,1,2)
        for k in range(3):
            u = [0]*9; u[8] = 1
            v = [0]*9; v[8] = 1
            w = [0]*9; w[k*3+8] = 1
            U.append(u); V.append(v); W.append(w)
        return U, V, W

    # 3. Build M_{3,5,6}
    # Partition I=3, J=5, K=6.
    # Let's use J=5 -> 3+2. K=6 -> 3+3.
    # Blocks:
    # 1. I=3, J=3, K=3 -> M_{3,3,3} (Rank 26)
    # 2. I=3, J=3, K=3 -> M_{3,3,3} (Rank 26)
    # 3. I=3, J=2, K=3 -> M_{3,2,3} (Rank 18, schoolbook)
    # 4. I=3, J=2, K=3 -> M_{3,2,3} (Rank 18, schoolbook)
    # Total Rank: 26 + 26 + 18 + 18 = 88.
    
    def build_block(n_b, m_b, p_b, offset_i, offset_j, offset_k):
        U = []
        V = []
        W = []
        # Schoolbook for this block
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u = [0]*(n*m); u[(offset_i+i)*m + (offset_j+j)] = 1
                    v = [0]*(m*p); v[(offset_j+j)*p + (offset_k+k)] = 1
                    w = [0]*(p*n); w[(offset_k+k)*n + (offset_i+i)] = 1
                    U.append(u); V.append(v); W.append(w)
        return U, V, W

    U, V, W = build_333()
    # Adjust indices for block (0,0,0)
    U_full = []
    V_full = []
    W_full = []
    for u, v, w in zip(U, V, W):
        U_full.append([0]*(n*m))
        V_full.append([0]*(m*p))
        W_full.append([0]*(p*n))
        for idx, val in enumerate(u):
            if val != 0:
                i = idx // 3
                j = idx % 3
                U_full[-1][i*m + j] = val
        for idx, val in enumerate(v):
            if val != 0:
                j = idx // 3
                k = idx % 3
                V_full[-1][j*p + k] = val
        for idx, val in enumerate(w):
            if val != 0:
                k = idx // 3
                i = idx % 3
                W_full[-1][k*n + i] = val
                
    # Add M_{3,3,3} at (0,3,3)
    U2, V2, W2 = build_333()
    for u, v, w in zip(U2, V2, W2):
        U_full.append([0]*(n*m))
        V_full.append([0]*(m*p))
        W_full.append([0]*(p*n))
        for idx, val in enumerate(u):
            if val != 0:
                i = idx // 3
                j = idx % 3
                U_full[-1][i*m + (3+j)] = val
        for idx, val in enumerate(v):
            if val != 0:
                j = idx // 3
                k = idx % 3
                V_full[-1][(3+j)*p + (3+k)] = val
        for idx, val in enumerate(w):
            if val != 0:
                k = idx // 3
                i = idx % 3
                W_full[-1][(3+k)*n + i] = val

    # Add M_{3,2,3} at (0,3,0) -> Schoolbook
    U3, V3, W3 = build_block(3, 2, 3, 0, 3, 0)
    U_full.extend(U3)
    V_full.extend(V3)
    W_full.extend(W3)
    
    # Add M_{3,2,3} at (0,3,3) -> Schoolbook
    U4, V4, W4 = build_block(3, 2, 3, 0, 3, 3)
    U_full.extend(U4)
    V_full.extend(V4)
    W_full.extend(W4)
    
    return {"U": U_full, "V": V_full, "W": W_full}
