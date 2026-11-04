"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

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

    def build_2x2x3_rank11():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3 (4 schoolbook + 7 Strassen)."""
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

    def try_find_2x2x3_lower_rank(max_rank=10, max_iters=5000):
        """Try to find a lower-rank decomposition for 2x2x3 using random search."""
        random.seed(42)
        np.random.seed(42)
        
        # The 2x2x3 multiplication tensor: T[i,j,k] for i in 0..1, j in 0..1, k in 0..2
        # A is 2x2, B is 2x3, C is 2x3
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # U has shape (4, rank) - indexed by (i,j)
        # V has shape (6, rank) - indexed by (j,k)
        # W has shape (6, rank) - indexed by (k,i)
        
        best_U, best_V, best_W = None, None, None
        best_rank = max_rank + 1
        
        for rank in range(8, max_rank + 1):
            for it in range(max_iters // max_rank):
                if time.time() > deadline - 5:
                    break
                    
                # Random initialization with values from {-1, -0.5, 0, 0.5, 1}
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(4, rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, rank))
                
                # Check if this decomposes the tensor
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            # U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :]
                            prod = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                            expected = 1.0 if (i == i and j == j and k == k) else 0.0
                            # Actually, the tensor element T[i,j,k] should be 1 if the 
                            # multiplication A[i,j]*B[j,k] contributes to C[i,k]
                            # The identity: C[i,k] = sum_j A[i,j] B[j,k]
                            # So the tensor is: T[i,j,k] = 1 for all valid (i,j,k)
                            # Wait no. The multiplication tensor has T[i,j,k] = 1 
                            # when the product A[i,j]*B[j,k] appears in C[i,k].
                            # For each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k].
                            # So T[i,j,k] = 1 for all i,j,k where this is valid.
                            # In our case, all combinations are valid, so T[i,j,k] = 1 for all.
                            expected = 1.0
                            if abs(prod - expected) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    best_rank = rank
                    break
            if best_rank <= rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None, None, None, max_rank + 1

    def build_4x4x6(U_sub, V_sub, W_sub):
        """Build 4x4x6 from 2x2x3 sub-decomposition."""
        n, m, p = 4, 4, 6
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

    # First try to find a lower-rank 2x2x3 decomposition
    U_sub, V_sub, W_sub, sub_rank = try_find_2x2x3_lower_rank(max_rank=10, max_iters=3000)
    
    if U_sub is None:
        # Fall back to rank-11
        U_sub, V_sub, W_sub = build_2x2x3_rank11()
        sub_rank = 11

    U, V, W = build_4x4x6(U_sub, V_sub, W_sub)

    # Verify the decomposition efficiently
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(np.dot(U_arr[i*m+j, :] * V_arr[j*p+k, :], W_arr[k*n+i, :]) - 1.0) > 1e-9:
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
