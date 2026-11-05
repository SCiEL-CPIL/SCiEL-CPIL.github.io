# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7 (local indices)
        Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        """
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def rank1_1x1x1():
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W

    def rank1_1x2x1():
        # 1x2x1 tensor: U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[0,1]=1
        # Wait, shape is (n_block, m_block, p_block).
        # U is (1, 2, 1) -> 2 entries. V is (2, 1, 1) -> 2 entries. W is (1, 1, 1) -> 1 entry? No.
        # Tensor T[i,j,k].
        # If n=1, m=2, p=1.
        # U[i,j,c], V[j,k,c], W[k,i,c].
        # Rank 1 means c=0.
        # U[0,0,0]=1, U[0,1,0]=1
        # V[0,0,0]=1, V[1,0,0]=1
        # W[0,0,0]=1
        U = [[1, 1]] # i=0, j=0,1
        V = [[1], [1]] # j=0,1, k=0
        W = [[1]] # k=0, i=0
        return U, V, W

    def rank1_1x1x2():
        # n=1, m=1, p=2
        U = [[1]]
        V = [[1, 1]] # j=0, k=0,1
        W = [[1, 1]] # k=0,1, i=0
        return U, V, W

    def rank1_2x1x1():
        # n=2, m=1, p=1
        U = [[1], [1]] # i=0,1, j=0
        V = [[1]]
        W = [[1, 1]] # k=0, i=0,1
        return U, V, W

    def rank1_3x1x1():
        # n=3, m=1, p=1
        U = [[1], [1], [1]]
        V = [[1]]
        W = [[1, 1, 1]]
        return U, V, W

    def rank2_1x2x2():
        # n=1, m=2, p=2. Rank 2.
        # We can decompose into two rank-1 terms.
        # T[0, j, k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c]
        # Let's use identity-like mapping.
        # c=0: covers j=0, k=0 and j=1, k=1? No, that's rank 2 for 2x2.
        # Standard rank 2 for 1x2x2 is just 2 rank 1s.
        # Term 1: j=0, k=0. U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1.
        # Term 2: j=1, k=1. U[0,1,1]=1, V[1,1,1]=1, W[1,0,1]=1.
        # This leaves T[0,0,1] and T[0,1,0] as 0. But the multiplication tensor is all 1s.
        # So this decomposition is WRONG for the all-ones tensor.
        # The tensor T[i,j,k] = 1 for all i,j,k in the block.
        # Can we represent a 1x2x2 all-ones tensor with rank 2?
        # T[j,k] = 1. Matrix is [[1,1],[1,1]]. Rank 1.
        # So 1x2x2 all-ones tensor has rank 1!
        # U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1.
        # Let's check: Sum_c U[0,j,c] V[j,k,c] W[k,0,c].
        # c=0: U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1. Product 1.
        # So rank 1 is sufficient for 1x2x2 all-ones.
        # My previous `rank1_1x2x1` was for 1x2x1.
        # For 1x2x2:
        U = [[1, 1]]
        V = [[1, 1], [1, 1]]
        W = [[1, 1]]
        return U, V, W

    def rank2_2x1x2():
        # n=2, m=1, p=2. All ones.
        # T[i,k] = 1. Matrix [[1,1],[1,1]]. Rank 1.
        U = [[1], [1]]
        V = [[1, 1]]
        W = [[1, 1]]
        return U, V, W

    def rank2_2x2x1():
        # n=2, m=2, p=1. All ones.
        # T[i,j] = 1. Matrix [[1,1],[1,1]]. Rank 1.
        U = [[1, 1], [1, 1]]
        V = [[1], [1]]
        W = [[1, 1]]
        return U, V, W

    def make_block(i_list, j_list, k_list, U_s, V_s, W_s):
        """
        Maps a local block decomposition to global columns.
        i_list, j_list, k_list: lists of local indices corresponding to the block.
        U_s, V_s, W_s: local factor matrices (lists of lists, rows are local indices, cols are rank).
        Returns U_cols, V_cols, W_cols: lists of global columns.
        """
        rank = len(U_s[0]) if U_s else 0
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Map local indices to global indices
        # i_list corresponds to rows of U_s
        # j_list corresponds to rows of V_s
        # k_list corresponds to rows of W_s (first index) and columns of U_s/V_s? 
        # No, U is (n, m, rank). Local U_s is (len(i_list), len(j_list), rank).
        # V is (m, p, rank). Local V_s is (len(j_list), len(k_list), rank).
        # W is (p, n, rank). Local W_s is (len(k_list), len(i_list), rank).
        
        for c in range(rank):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            
            for li, i in enumerate(i_list):
                for lj, j in enumerate(j_list):
                    U_col[i * m + j] = U_s[li * len(j_list) + lj][c]
                    
            for lj, j in enumerate(j_list):
                for lk, k in enumerate(k_list):
                    V_col[j * p + k] = V_s[lj * len(k_list) + lk][c]
                    
            for lk, k in enumerate(k_list):
                for li, i in enumerate(i_list):
                    W_col[k * n + i] = W_s[lk * len(i_list) + li][c]
                    
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            
        return U_cols, V_cols, W_cols

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    def build_decomposition(strategy_name):
        """Build a decomposition based on a strategy. Returns (U_cols, V_cols, W_cols)."""
        U_cols = []
        V_cols = []
        W_cols = []

        if strategy_name == "G":
            # Strategy G: Mixed - use Strassen for 2x2x2 where beneficial, rank-1/2 elsewhere
            # For i in {0, 1}, j in {0,1}, k in {0,1}: Strassen (7)
            # For i in {0, 1}, j in {2,3}, k in {0,1}: Strassen (7)
            # For i in {0, 1}, j in {0,1}, k in {2,3}: Strassen (7)
            # For i in {0, 1}, j in {2,3}, k in {2,3}: Strassen (7)
            # For i in {0, 1}, k=4, j in {0,1}: rank 1 (1x2x1 is rank 1)
            # For i in {0, 1}, k=4, j in {2,3}: rank 1
            # For i=2, j in {0,1}, k in {0,1}: rank 1 (1x2x2 is rank 1)
            # For i=2, j in {2,3}, k in {0,1}: rank 1
            # For i=2, j in {0,1}, k in {2,3}: rank 1
            # For i=2, j in {2,3}, k in {2,3}: rank 1
            # For i=2, k=4, j in {0,1}: rank 1
            # For i=2, k=4, j in {2,3}: rank 1
            # Total: 4*7 + 4*1 + 4*1 + 2*1 = 28 + 4 + 4 + 2 = 38
            
            for i in [0, 1]:
                for j_off in [0, 2]:
                    for k_off in [0, 2]:
                        U_s, V_s, W_s = strassen_2x2x2()
                        uc, vc, wc = make_block([i], [j_off, j_off+1], [k_off, k_off+1], U_s, V_s, W_s)
                        U_cols.extend(uc)
                        V_cols.extend(vc)
                        W_cols.extend(wc)
                for j_off in [0, 2]:
                    U_s, V_s, W_s = rank1_1x2x1()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [4], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
            for j_off in [0, 2]:
                for k_off in [0, 2]:
                    U_s, V_s, W_s = rank2_1x2x2() # Which is actually rank 1
                    uc, vc, wc = make_block([2], [j_off, j_off+1], [k_off, k_off+1], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                U_s, V_s, W_s = rank1_1x2x1()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [4], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)

        elif strategy_name == "L":
            # Strategy L: Hybrid - use Strassen for 2x2x2, but also try 2x2x1 (rank 2) blocks
            # Same as G essentially in terms of rank count if rank2_1x2x2 is rank 1.
            for i in [0, 1]:
                for j_off in [0, 2]:
                    for k_off in [0, 2]:
                        U_s, V_s, W_s = strassen_2x2x2()
                        uc, vc, wc = make_block([i], [j_off, j_off+1], [k_off, k_off+1], U_s, V_s, W_s)
                        U_cols.extend(uc)
                        V_cols.extend(vc)
                        W_cols.extend(wc)
                    U_s, V_s, W_s = rank1_1x2x1()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [4], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
            for j_off in [0, 2]:
                for k_off in [0, 2]:
                    U_s, V_s, W_s = rank2_1x2x2()
                    uc, vc, wc = make_block([2], [j_off, j_off+1], [k_off, k_off+1], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                U_s, V_s, W_s = rank1_1x2x1()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [4], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)

        elif strategy_name == "M":
            # Strategy M: Try 2x2x3 blocks - decompose 2x2x3 as 2x2x2 + 2x2x1
            # 2x2x2: Strassen (7), 2x2x1: rank 1 (since 2x2x1 all ones is rank 1)
            # For i in {0,1}, j in {0,1}: 2x2x2 (7) + 2x2x1 (1) = 8
            # For i in {0,1}, j in {2,3}: 2x2x2 (7) + 2x2x1 (1) = 8
            # For i=2, j in {0,1}: 1x2x2 (1) + 1x2x1 (1) = 2
            # For i=2, j in {2,3}: 1x2x2 (1) + 1x2x1 (1) = 2
            # Total: 2*8 + 2*2 = 20? 
            # Wait, 2x2x2 is 7. 2x2x1 is 1.
            # Let's verify 2x2x1 rank.
            # T[i,j,0]=1. Matrix [[1,1],[1,1]] rank 1.
            # So 2x2x3 rank is 7+1=8.
            # 1x2x3: 1x2x2 (rank 1) + 1x2x1 (rank 1) = 2.
            # Total: 4*7 (for i=0,1, j=0,1,2,3, k=0,1,2,3) ? No.
            # i=0,1: j=0,1 (2x2x3), j=2,3 (2x2x3).
            # i=2: j=0,1 (1x2x3), j=2,3 (1x2x3).
            # i=0,1, j=0,1: 7+1=8
            # i=0,1, j=2,3: 7+1=8
            # i=2, j=0,1: 1+1=2
            # i=2, j=2,3: 1+1=2
            # Total: 8+8+2+2 = 20.
            # This is significantly better than 38.
            
            for i in [0, 1]:
                for j_off in [0, 2]:
                    # 2x2x2 for k=0,1
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    # 2x2x2 for k=2,3
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    # 2x2x1 for k=4
                    U_s, V_s, W_s = rank2_2x2x1() # Rank 1
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [4], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
            for j_off in [0, 2]:
                # 1x2x2 for k=0,1
                U_s, V_s, W_s = rank2_1x2x2() # Rank 1
                uc, vc, wc = make_block([2], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                # 1x2x2 for k=2,3
                U_s, V_s, W_s = rank2_1x2x2()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                # 1x2x1 for k=4
                U_s, V_s, W_s = rank1_1x2x1()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [4], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)

        elif strategy_name == "N":
            # Strategy N: Same as M
            for i in [0, 1]:
                for j_off in [0, 2]:
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    U_s, V_s, W_s = rank2_2x2x1()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [4], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
            for j_off in [0, 2]:
                U_s, V_s, W_s = rank2_1x2x2()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                U_s, V_s, W_s = rank2_1x2x2()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                U_s, V_s, W_s = rank1_1x2x1()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [4], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)

        elif strategy_name == "O":
            # Strategy O: Same as M
            for i in [0, 1]:
                for j_off in [0, 2]:
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    U_s, V_s, W_s = strassen_2x2x2()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
                    U_s, V_s, W_s = rank2_2x2x1()
                    uc, vc, wc = make_block([i], [j_off, j_off+1], [4], U_s, V_s, W_s)
                    U_cols.extend(uc)
                    V_cols.extend(vc)
                    W_cols.extend(wc)
            for j_off in [0, 2]:
                U_s, V_s, W_s = rank2_1x2x2()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [0, 1], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                U_s, V_s, W_s = rank2_1x2x2()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [2, 3], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)
                U_s, V_s, W_s = rank1_1x2x1()
                uc, vc, wc = make_block([2], [j_off, j_off+1], [4], U_s, V_s, W_s)
                U_cols.extend(uc)
                V_cols.extend(vc)
                W_cols.extend(wc)

        return U_cols, V_cols, W_cols

    # Try all strategies
    strategies = ["G", "L", "M", "N", "O"]
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        U_cols, V_cols, W_cols = build_decomposition(strat)
        rank = len(U_cols)
        if rank >= best_rank:
            continue
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        if verify_fast(U_mat, V_mat, W_mat, n, m, p):
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
