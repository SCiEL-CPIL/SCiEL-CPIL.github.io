"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s evaluator kill
    n, m, p = DIMENSIONS

    # Default: schoolbook rank = n*m*p = 36
    def schoolbook():
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

    best = schoolbook()

    try:
        # Strategy: Block Decomposition
        # 1. Construct a rank-11 decomposition for a 2x3x3 sub-problem.
        #    - Use 7 Strassen terms for the 2x2x2 part (j in {0,1}, k in {0,1}).
        #    - Use 4 schoolbook terms for the remaining part (j=2, k in {0,1}).
        # 2. Verify the 2x3x3 decomposition.
        # 3. Instantiate two independent copies of this block to cover k=0..5.
        #    Total rank = 11 + 11 = 22.
        
        # Dimensions for the sub-block
        n_sub = 2
        m_sub = 3
        p_sub = 3
        rank_sub = 11
        
        U_sub = np.zeros((n_sub * m_sub, rank_sub))
        V_sub = np.zeros((m_sub * p_sub, rank_sub))
        W_sub = np.zeros((p_sub * n_sub, rank_sub))

        # Strassen terms for 2x2x2
        # Indices for U: i*2 + j (i in 0..1, j in 0..1)
        # Indices for V: j*2 + k (j in 0..1, k in 0..1)
        # Indices for W: k*2 + i (k in 0..1, i in 0..1)
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # p1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # p2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # p3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # p4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # p5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # p6
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]), # p7
        ]

        # Fill U_sub, V_sub, W_sub with Strassen terms for j in {0,1}
        for t in range(7):
            ut, vt, wt = strassen_terms[t]
            # U_sub indices: i * m_sub + j. Here j is 0 or 1.
            # ut is indexed by [i*2 + j] for i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U_sub[i * m_sub + j, t] = ut[i * 2 + j]
            
            # V_sub indices: j * p_sub + k. Here j is 0 or 1.
            # vt is indexed by [j*2 + k] for j in 0..1, k in 0..1
            for j in range(2):
                for k in range(2):
                    V_sub[j * p_sub + k, t] = vt[j * 2 + k]
            
            # W_sub indices: k * n_sub + i. Here k is 0 or 1.
            # wt is indexed by [k*2 + i] for k in 0..1, i in 0..1
            for k in range(2):
                for i in range(2):
                    W_sub[k * n_sub + i, t] = wt[k * 2 + i]

        # Fill U_sub, V_sub, W_sub with Schoolbook terms for j=2
        # These terms handle the multiplication for the 3rd row of A (j=2)
        # We need to cover k in {0, 1} for the sub-block.
        # Term t=7: i=0, k=0
        # Term t=8: i=0, k=1
        # Term t=9: i=1, k=0
        # Term t=10: i=1, k=1
        schoolbook_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, (i_val, k_val) in enumerate(schoolbook_indices):
            t = 7 + idx
            # U_sub[i * m_sub + j] where j=2
            U_sub[i_val * m_sub + 2, t] = 1
            # V_sub[j * p_sub + k] where j=2
            V_sub[2 * p_sub + k_val, t] = 1
            # W_sub[k * n_sub + i]
            W_sub[k_val * n_sub + i_val, t] = 1

        # Verify the 2x3x3 sub-decomposition
        def verify_sub(Uf, Vf, Wf, n, m, p, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(rank):
                            s += Uf[i * m + j, t] * Vf[j * p + k, t] * Wf[k * n + i, t]
                        if abs(s - 1) > 1e-9:
                            return False
            return True

        if not verify_sub(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, rank_sub):
            # If sub-decomposition fails, fallback
            return best

        # Construct full 2x3x6 decomposition
        # We have 2 blocks of size 3 in the k-dimension.
        # Block 0: k=0,1,2
        # Block 1: k=3,4,5
        total_rank = 2 * rank_sub
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))

        for block_idx in range(2):
            k_offset = block_idx * p_sub
            rank_offset = block_idx * rank_sub
            
            # Copy U_sub to U_full
            # U indices are the same (i, j)
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j, rank_offset : rank_offset + rank_sub] = U_sub[i * m_sub + j]
            
            # Copy V_sub to V_full
            # V indices are (j, k). k in sub-block is 0..2. k in full is k_offset + k_sub.
            for j in range(m):
                for k_sub in range(p_sub):
                    k_full = k_offset + k_sub
                    V_full[j * p + k_full, rank_offset : rank_offset + rank_sub] = V_sub[j * p_sub + k_sub]
            
            # Copy W_sub to W_full
            # W indices are (k, i). k in sub-block is 0..2. k in full is k_offset + k_sub.
            for k_sub in range(p_sub):
                k_full = k_offset + k_sub
                for i in range(n):
                    W_full[k_full * n + i, rank_offset : rank_offset + rank_sub] = W_sub[k_sub * n_sub + i]

        # Verify full decomposition
        def verify_full(Uf, Vf, Wf, n, m, p, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(rank):
                            s += Uf[i * m + j, t] * Vf[j * p + k, t] * Wf[k * n + i, t]
                        if abs(s - 1) > 1e-9:
                            return False
            return True

        if not verify_full(U_full, V_full, W_full, n, m, p, total_rank):
            return best

        def to_clean(mat):
            result = []
            for row in mat:
                clean_row = []
                for x in row:
                    if abs(x - round(x)) < 1e-9:
                        clean_row.append(int(round(x)))
                    elif abs(2 * x - round(2 * x)) < 1e-9:
                        clean_row.append(round(2 * x) / 2)
                    else:
                        clean_row.append(x)
                result.append(clean_row)
            return result

        return {"U": to_clean(U_full.tolist()), "V": to_clean(V_full.tolist()), "W": to_clean(W_full.tolist())}

    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
