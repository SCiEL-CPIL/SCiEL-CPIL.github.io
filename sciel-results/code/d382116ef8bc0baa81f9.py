"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x4x6_sub():
        """Rank-12 decomposition for 2x4 * 4x6 -> 2x6.
        
        A is 2x4, split into A1 (2x2) and A2 (2x2).
        B is 4x6, split into B1 (2x6) and B2 (2x6).
        C is 2x6, split into C1 (2x6) and C2 (2x6).
        
        C1 = A1*B1: rank 7 (Strassen)
        C2 = A2*B2: rank 5 (schoolbook, since 2x2 * 2x6)
        Total: 12
        """
        rank = 12
        U = [[0]*rank for _ in range(8)]   # 2*4 = 8 rows (i*4+j)
        V = [[0]*rank for _ in range(24)]  # 4*6 = 24 rows (j*6+k)
        W = [[0]*rank for _ in range(12)]  # 6*2 = 12 rows (k*2+i)

        # Part 1: A1*B1 -> C1 (Strassen, rank 7)
        # A1 is rows 0-1, cols 0-1 of A: U indices i*4+j for i in 0..1, j in 0..1
        # B1 is rows 0-1, cols 0-5 of B: V indices j*6+k for j in 0..1, k in 0..5
        # C1 is rows 0-1, cols 0-5 of C: W indices k*2+i for k in 0..5, i in 0..1

        # Strassen terms for 2x2 * 2x2, but B is 2x6 (only use cols 0-1)
        # Actually, B1 is 2x6, so we need 2x2 * 2x6 -> 2x6
        # Let me reconsider: A1 is 2x2, B1 is 2x6
        # We can split B1 into B1a (2x2) and B1b (2x4)
        # A1*B1a: Strassen rank 7
        # A1*B1b: schoolbook rank 8 (2*2*4/2... no, 2*4 = 8)
        # That gives 15, not 7.
        
        # Alternative: just use schoolbook for the whole 2x2 * 2x6
        # Rank = 2*2*6/2... no. Schoolbook rank for 2x2 * 2x6 is 2*2*6 = 24? No.
        # Schoolbook: each C[i,k] = sum_j A[i,j]*B[j,k], so 2*2*6 = 24 terms.
        # That's too many.
        
        # Let me reconsider the whole approach.
        # 2x4 * 4x6: 
        # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x2
        # Split B = [B1; B2] where B1 is 2x6, B2 is 2x6
        # C = A1*B1 + A2*B2
        # A1*B1: 2x2 * 2x6 -> 2x6, schoolbook rank = 2*2*6 = 24? No.
        # Actually schoolbook for (n x m) * (m x p) has rank n*m*p only if we do it naively.
        # For 2x2 * 2x6: each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # That's 2*2*6 = 24 scalar multiplications, but as a tensor rank...
        # The tensor T_{i,j,k} = delta_{i,j} for the 2x2 part... 
        # Actually the multiplication tensor for (2x2)*(2x6) has rank at most 2*2*6=24
        # but can be done with Strassen on the 2x2 part: 
        # Use Strassen to compute A1*B1a (2x2 * 2x2 -> 2x2) with rank 7
        # And schoolbook for A1*B1b (2x2 * 2x4 -> 2x4) with rank 2*2*4=16? No.
        
        # I think I'm overcomplicating this. Let me just use the full schoolbook 
        # for the 2x4 * 4x6 sub-problem and see what rank that is.
        # Schoolbook rank for (2x4)*(4x6) = 2*4*6 = 48. That's worse.
        
        # Let me try a different split.
        # Actually, let me just implement the full 4x4x6 with a different strategy.
        
        # New approach: Use the fact that we can split along different dimensions.
        # Let me try: split B into two 4x3 blocks.
        # C = A*B1 + A*B2 where B1 is 4x3, B2 is 4x3
        # A*B1: 4x4 * 4x3 -> 4x3
        # A*B2: 4x4 * 4x3 -> 4x3
        # For 4x4 * 4x3: split A into 2x2 blocks, B into 2x3 blocks
        # Use Strassen: 7 sub-problems of 2x2 * 2x3
        # 2x2 * 2x3: split B into 2x1 and 2x2
        #   2x2 * 2x1: schoolbook rank 4
        #   2x2 * 2x2: Strassen rank 7
        #   Total: 11
        # So 4x4 * 4x3 has rank 7*11 = 77
        # Total for 4x4 * 4x6: 2*77 = 154. Worse.
        
        # I think the original rank-77 approach is hard to beat with simple block splits.
        # Let me just return the original construction but verify it works.
        return None

    # Let me just use the original approach but make sure it's correct.
    # I'll implement the rank-77 solution and verify it.

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

        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base + c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base + c] += W2[r][c]

        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6

        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub

            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff

                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff

                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :] * V_arr[j*p+k, :], W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
