"""Rank-88 tensor decomposition for 3x4x8 matrix multiplication."""
import time

N, M, P = 3, 4, 8
RANK = 88

# Pre-compute non-zero entries for Strassen factors
# Format: list of (local_index, value)
# Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

strassen_U_nonzero = [
    [(0, 1), (3, 1)],      # M1
    [(2, 1), (3, 1)],      # M2
    [(0, 1)],              # M3
    [(3, 1)],              # M4
    [(0, 1), (1, 1)],      # M5
    [(0, -1), (2, 1)],     # M6
    [(1, 1), (3, -1)]      # M7
]

strassen_V_nonzero = [
    [(0, 1), (3, 1)],      # M1
    [(0, 1)],              # M2
    [(1, 1), (3, -1)],     # M3
    [(0, -1), (2, 1)],     # M4
    [(3, 1)],              # M5
    [(0, 1), (1, 1)],      # M6
    [(2, 1), (3, 1)]       # M7
]

strassen_W_nonzero = [
    [(0, 1), (3, 1)],      # M1
    [(2, 1), (3, -1)],     # M2
    [(1, 1), (3, 1)],      # M3
    [(0, 1), (2, 1)],      # M4
    [(0, -1), (1, 1)],     # M5
    [(3, 1)],              # M6
    [(0, 1)]               # M7
]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    U = [[0] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8 (32 terms)
    for j in range(M):
        for k in range(P):
            U[j, col] = 1          # A[0,j] -> index 0*M+j = j
            V[j * P + k, col] = 1  # B[j,k]
            W[k * N + 0, col] = 1  # C[0,k] -> index k*N+0
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    # Part 2: i=1,2 via 8 Strassen 2x2x2 blocks
    # j_blocks: [0,1] and [2,3]; k_blocks: [0,1],[2,3],[4,5],[6,7]
    
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                # Process U
                for l, val in strassen_U_nonzero[t]:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    gj = js + lj
                    U[gi * M + gj, col] += val
                
                # Process V
                for l, val in strassen_V_nonzero[t]:
                    lj = l // 2
                    lk = l % 2
                    gj = js + lj
                    gk = ks + lk
                    V[gj * P + gk, col] += val
                
                # Process W
                for l, val in strassen_W_nonzero[t]:
                    li = l // 2
                    lk = l % 2
                    gi = 1 + li
                    gk = ks + lk
                    W[gk * N + gi, col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
