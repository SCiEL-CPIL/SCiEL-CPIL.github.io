"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 3, 8

    # Strategy: Split 3x8 into two 3x4 blocks, use rank-10 for each 2x3x4.
    n_sub, m_sub, p_sub = 2, 3, 4
    rank_sub = 10

    # Build rank-10 decomposition for 2x3x4 via ALS
    np.random.seed(123)
    U_sub = np.random.randn(n_sub * m_sub, rank_sub) * 0.5
    V_sub = np.random.randn(m_sub * p_sub, rank_sub) * 0.5
    W_sub = np.random.randn(p_sub * n_sub, rank_sub) * 0.5

    def solve_factor(size, other1, other2, n_s, m_s, p_s, factor_type):
        """Solve for one factor given the other two."""
        num_constraints = n_s * m_s * p_s
        num_unknowns = size * rank_sub
        A = np.zeros((num_constraints, num_unknowns))
        b_vec = np.ones(num_constraints)
        
        idx = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    a = i * m_s + j
                    b = j * p_s + k
                    c_idx = k * n_s + i
                    for c in range(rank_sub):
                        if factor_type == 'U':
                            col = a * rank_sub + c
                            A[idx, col] = other1[b, c] * other2[c_idx, c]
                        elif factor_type == 'V':
                            col = b * rank_sub + c
                            A[idx, col] = other1[a, c] * other2[c_idx, c]
                        else:  # W
                            col = c_idx * rank_sub + c
                            A[idx, col] = other1[a, c] * other2[b, c]
                    idx += 1
        
        sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        return sol.reshape(size, rank_sub)

    for it in range(1000):
        if time.time() > deadline - 20:
            break
        U_sub = solve_factor(n_sub * m_sub, V_sub, W_sub, n_sub, m_sub, p_sub, 'U')
        V_sub = solve_factor(m_sub * p_sub, U_sub, W_sub, n_sub, m_sub, p_sub, 'V')
        W_sub = solve_factor(p_sub * n_sub, U_sub, V_sub, n_sub, m_sub, p_sub, 'W')
        
        if it % 100 == 0:
            err = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        s = np.dot(U_sub[i*m_sub+j], V_sub[j*p_sub+k] * W_sub[k*n_sub+i])
                        err = max(err, abs(s - 1.0))
            if err < 1e-9:
                break

    # Check if 2x3x4 decomposition is valid
    err = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                s = np.dot(U_sub[i*m_sub+j], V_sub[j*p_sub+k] * W_sub[k*n_sub+i])
                err = max(err, abs(s - 1.0))

    if err > 1e-6:
        # Fallback: schoolbook
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

    # Build full 2x3x8 from two 2x3x4 blocks
    total_rank = 2 * rank_sub
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    for c in range(rank_sub):
        # Block 1: k=0..3
        U_full[:, c] = U_sub[:, c]
        for j in range(m_sub):
            for k in range(p_sub):
                V_full[j*p+k, c] = V_sub[j*p_sub+k, c]
        for k in range(p_sub):
            for i in range(n_sub):
                W_full[k*n+i, c] = W_sub[k*n_sub+i, c]
        
        # Block 2: k=4..7
        U_full[:, c + rank_sub] = U_sub[:, c]
        for j in range(m_sub):
            for k in range(p_sub):
                V_full[j*p+(k+p_sub), c + rank_sub] = V_sub[j*p_sub+k, c]
        for k in range(p_sub):
            for i in range(n_sub):
                W_full[(k+p_sub)*n+i, c + rank_sub] = W_sub[k*n_sub+i, c]

    # Verify
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_full[i*m+j], V_full[j*p+k] * W_full[k*n+i])
                err = max(err, abs(s - 1.0))

    if err > 1e-6:
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

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
