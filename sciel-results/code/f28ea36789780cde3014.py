"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # M1 = (a00+a11)(b00+b11)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2 = (a10+a11)b00
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        # M3 = a00(b10-b00)
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        # M4 = a11(b01-b10)
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5 = (a00+a01)b11
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6 = (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7 = (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        
        Splits B (2x3) into B_12 (2x2, cols 0-1) and B_3 (2x1, col 2).
        Uses Strassen for A * B_12 (rank 7) and schoolbook for A * B_3 (rank 4).
        Total rank: 11.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # Part 1: A * B_12 using Strassen (cols 0-1 of B, cols 0-1 of C)
        U2, V2, W2 = build_2x2x2_strassen()
        # V2 has 4 rows (2x2). We map them to V rows for cols 0,1 of the 2x3 block.
        # V index in 2x3 block: row j (0-1), col k (0-2).
        # For B_12, k is 0 or 1. So V rows are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1) in local block coords?
        # Wait, V in build_2x2x2 is indexed by j*2+k.
        # In the 2x3 block, V index is j*3+k.
        # So for B_12 (k=0,1):
        # j=0,k=0 -> V[0]
        # j=0,k=1 -> V[1]
        # j=1,k=0 -> V[3]
        # j=1,k=1 -> V[4]
        v2_map = {0:0, 1:1, 2:3, 3:4}
        # W2 has 4 rows (2x2). Map to W rows for cols 0,1 of the 2x3 block.
        # W index in 2x3 block: row i (0-1), col k (0-2).
        # W is indexed by i*? No, W in tensor is C_flat.
        # C is 2x3. C_flat index is k*n + i? No, task says W[k*n+i].
        # So for C (2x3), index is k*2 + i.
        # For C_12 (k=0,1):
        # k=0,i=0 -> W[0]
        # k=0,i=1 -> W[1]
        # k=1,i=0 -> W[2]
        # k=1,i=1 -> W[3]
        w2_map = {0:0, 1:1, 2:2, 3:3}
        
        for c in range(7):
            for r in range(4):
                if U2[r][c]: U[r][c] += U2[r][c]
            for r in range(4):
                if V2[r][c]: V[v2_map[r]][c] += V2[r][c]
            for r in range(4):
                if W2[r][c]: W[w2_map[r]][c] += W2[r][c]
        
        # Part 2: A * B_3 using Schoolbook (col 2 of B, col 2 of C)
        # B_3 is 2x1. C_3 is 2x1.
        # Schoolbook for 2x2 * 2x1 has rank 4.
        # c0: a00*b0
        # c1: a01*b1
        # c2: a10*b0
        # c3: a11*b1
        base = 7
        U1, V1, W1 = build_2x2x1()
        # V1 has 2 rows (2x1). Map to V rows for col 2 of the 2x3 block.
        # j=0,k=2 -> V[2]
        # j=1,k=2 -> V[5]
        v1_map = {0:2, 1:5}
        # W1 has 2 rows (2x1). Map to W rows for col 2 of the 2x3 block.
        # k=2,i=0 -> W[4]
        # k=2,i=1 -> W[5]
        w1_map = {0:4, 1:5}
        
        for c in range(4):
            col_idx = base + c
            for r in range(4):
                if U1[r][c]: U[r][col_idx] += U1[r][c]
            for r in range(2):
                if V1[r][c]: V[v1_map[r]][col_idx] += V1[r][c]
            for r in range(2):
                if W1[r][c]: W[w1_map[r]][col_idx] += W1[r][c]
        
        return U, V, W

    def build_4x4x6_strassen():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # Outer Strassen steps for 2x2 blocks
        # A is 4x4, split into 2x2 blocks.
        # B is 4x6, split into 2x3 blocks.
        # C is 4x6, split into 2x3 blocks.
        
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        
        # Offsets for 2x2 blocks of A (4x4)
        # Block 0: rows 0-1, cols 0-1
        # Block 1: rows 0-1, cols 2-3
        # Block 2: rows 2-3, cols 0-1
        # Block 3: rows 2-3, cols 2-3
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        
        # Offsets for 2x3 blocks of B (4x6)
        # Block 0: rows 0-1, cols 0-2
        # Block 1: rows 0-1, cols 3-5
        # Block 2: rows 2-3, cols 0-2
        # Block 3: rows 2-3, cols 3-5
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        
        # Offsets for 2x3 blocks of C (4x6)
        # Block 0: rows 0-1, cols 0-2
        # Block 1: rows 0-1, cols 3-5
        # Block 2: rows 2-3, cols 0-2
        # Block 3: rows 2-3, cols 3-5
        w_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                # Process U
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_l, j_l = u_local // 2, u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        U[(i_l+ro)*m + (j_l+co)][base_col+c] += u_val*coeff
                
                # Process V
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_l, k_l = v_local // 3, v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        V[(j_l+ro)*p + (k_l+co)][base_col+c] += v_val*coeff
                
                # Process W
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_l, i_l = w_local // 2, w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        W[(k_l+ro)*n + (i_l+co)][base_col+c] += w_val*coeff
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Primary: Strassen-based rank 77
    U, V, W = build_4x4x6_strassen()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
