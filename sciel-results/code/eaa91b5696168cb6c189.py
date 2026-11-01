"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

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

    # Verification helper
    def verify_decomp(Um, Vm, Wm, n_local, m_local, p_local, rank_local):
        """Check if Um, Vm, Wm correctly decompose the n_local x m_local x p_local matmul tensor."""
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    s = 0.0
                    for t in range(rank_local):
                        s += Um[i * m_local + j][t] * Vm[j * p_local + k][t] * Wm[k * n_local + i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Convert to clean integer/half-integer representation
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

    # Strategy: split p=6 into three blocks of size 2. For each block, use:
    #   - 7 Strassen terms for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in block)
    #   - 4 schoolbook terms for j=2 (i in {0,1}, k in block)
    # Total per block: 11. Total rank: 3 * 11 = 33.

    try:
        rank = 33
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        # Strassen terms for 2x2x2 (standard formulation):
        # p1 = (A00+A11)*(B00+B11)
        # p2 = (A10+A11)*B00
        # p3 = A00*(B01-B11)
        # p4 = A11*(B10-B00)
        # p5 = (A00+A01)*B11
        # p6 = (A10-A00)*(B00+B01)
        # p7 = (A01-A11)*(B10+B11)
        #
        # C00 = p1 + p4 - p5 - p7
        # C01 = p3 + p5
        # C10 = p2 + p7
        # C11 = p1 - p2 + p3 + p6
        #
        # Each term as (u_vec, v_vec, w_vec) for the 2x2x2 subproblem:
        # u_vec[i*2+j] for i,j in {0,1}
        # v_vec[j*2+k] for j,k in {0,1}
        # w_vec[k*2+i] for k,i in {0,1}

        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # p1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # p2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # p3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # p4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # p5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # p6
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]), # p7
        ]

        # Build one 2x3x2 block (11 terms)
        U_block = [[0.0] * 11 for _ in range(6)]
        V_block = [[0.0] * 11 for _ in range(6)]
        W_block = [[0.0] * 11 for _ in range(4)]

        # Place 7 Strassen terms (for j in {0,1})
        for t in range(7):
            ut, vt, wt = strassen_terms[t]
            for i in range(2):
                for j in range(2):
                    U_block[i * 3 + j][t] = ut[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    V_block[j * 2 + k][t] = vt[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_block[k * 2 + i][t] = wt[k * 2 + i]

        # Place 4 schoolbook terms (for j=2)
        for di, (i_val, k_val) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            t = 7 + di
            U_block[i_val * 3 + 2][t] = 1
            V_block[2 * 2 + k_val][t] = 1
            W_block[k_val * 2 + i_val][t] = 1

        # Verify the block decomposition locally
        if not verify_decomp(U_block, V_block, W_block, 2, 3, 2, 11):
            # Block verification failed; fall back to schoolbook
            return best

        # Place 3 blocks into full U, V, W
        for block_idx in range(3):
            k_off = block_idx * 2
            for local_i in range(2):
                for local_j in range(3):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        U[local_i * 3 + local_j][full_t] = U_block[local_i * 3 + local_j][t]
            for local_j in range(3):
                for local_k in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        V[local_j * 2 + local_k][full_t] = V_block[local_j * 2 + local_k][t]
            for local_k in range(2):
                for local_i in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        W[(k_off + local_k) * 2 + local_i][full_t] = W_block[local_k * 2 + local_i][t]

        # Full verification
        if verify_decomp(U, V, W, n, m, p, rank):
            return {"U": to_clean(U), "V": to_clean(V), "W": to_clean(W)}

    except Exception:
        pass

    # Try to find a lower rank by attempting a 32-term decomposition
    # Strategy: attempt to merge one term across blocks
    try:
        if time.time() > deadline - 30:
            return best
        
        # Try rank 32: use 11 + 11 + 10 terms
        # This is speculative; try a simple perturbation approach
        rank = 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # First two blocks use 11 terms each
        for block_idx in range(2):
            k_off = block_idx * 2
            for local_i in range(2):
                for local_j in range(3):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        U[local_i * 3 + local_j][full_t] = U_block[local_i * 3 + local_j][t]
            for local_j in range(3):
                for local_k in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        V[local_j * 2 + local_k][full_t] = V_block[local_j * 2 + local_k][t]
            for local_k in range(2):
                for local_i in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        W[(k_off + local_k) * 2 + local_i][full_t] = W_block[local_k * 2 + local_i][t]
        
        # Third block uses 10 terms - attempt to share one term
        # This is a heuristic; may not work
        k_off = 4
        for local_i in range(2):
            for local_j in range(3):
                for t in range(10):
                    full_t = 22 + t
                    U[local_i * 3 + local_j][full_t] = U_block[local_i * 3 + local_j][t]
        for local_j in range(3):
            for local_k in range(2):
                for t in range(10):
                    full_t = 22 + t
                    V[local_j * 2 + local_k][full_t] = V_block[local_j * 2 + local_k][t]
        for local_k in range(2):
            for local_i in range(2):
                for t in range(10):
                    full_t = 22 + t
                    W[(k_off + local_k) * 2 + local_i][full_t] = W_block[local_k * 2 + local_i][t]
        
        # This won't work as-is since we're missing the 11th term
        # Fall back if verification fails
        if not verify_decomp(U, V, W, n, m, p, rank):
            return best
        else:
            return {"U": to_clean(U), "V": to_clean(V), "W": to_clean(W)}
            
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
