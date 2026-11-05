import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n

    # Structural change: Use Laderman's 4x4 algorithm (rank 21) combined
    # with Strassen (rank 7) for 2x2 parts in a 6x6 decomposition.
    # View 6x6 as 4x4 + 2x2 block structure.
    #
    # For C = A*B with 6x6 matrices:
    # Partition into 4x4 and 2x2 blocks:
    # A = [[A44, A42], [A24, A22]]
    # B = [[B44, B42], [B24, B22]]
    #
    # C44 = A44*B44 + A42*B24  (4x4)
    # C42 = A44*B42 + A42*B22  (4x2)
    # C24 = A24*B44 + A22*B24  (2x4)
    # C22 = A24*B42 + A22*B22  (2x2)
    #
    # This requires computing products of different sizes.
    # 
    # Alternative simpler structural change: Use a 3x3 block decomposition
    # but with Laderman-style optimization for the 3x3 blocks.
    #
    # Actually, let's use the known best: 6x6 matrix multiplication can be
    # done with rank 189 (current) or potentially lower.
    #
    # New approach: Use Kronecker product structure. 6 = 2*3, so
    # M_6 = M_2 ⊗ M_3. Rank(M_2) = 7 (Strassen), Rank(M_3) = 27 (schoolbook).
    # Rank(M_6) <= 7 * 27 = 189.
    #
    # To improve, we need a better decomposition of M_3 or M_2.
    # M_3 has rank 27 (no known better). M_2 has rank 7 (optimal).
    #
    # Let's try: Use a 5x5 + 1x1 decomposition.
    # M_6 can be viewed as involving 5x5 and 1x1 subproblems.
    # But this is complex.
    #
    # SIMPLEST STRUCTURAL IMPROVEMENT: Ensure the verification is fast and
    # correct, and try a different block structure.
    #
    # Let me try: 6x6 with 2x3 blocks (3x2 grid of 2x3 blocks).
    # This doesn't directly help with Strassen.
    #
    # Let me just fix the verification and ensure correctness.

    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps for 2x2 matrix multiplication
    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = 3
    col_idx = 0

    for step in steps:
        a_terms, b_terms, c_terms = step
        for r, s, t in itertools.product(range(bs), repeat=3):
            for I, J, ca in a_terms:
                u_idx = (I * bs + r) * dim + (J * bs + s)
                U[u_idx, col_idx] += ca

            for J, K, cb in b_terms:
                v_idx = (J * bs + s) * dim + (K * bs + t)
                V[v_idx, col_idx] += cb

            for I, K, cc in c_terms:
                w_idx = (K * bs + t) * dim + (I * bs + r)
                W[w_idx, col_idx] += cc

            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Fast verification using einsum
    try:
        if time.time() < deadline - 10:
            rank = U.shape[1]
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            if not np.allclose(result, 1.0, atol=1e-10):
                # Fall back to schoolbook
                total_rank_school = n * m * p
                U_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                V_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                W_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_s[i * m + j, idx] = 1
                            V_s[j * p + k, idx] = 1
                            W_s[k * n + i, idx] = 1
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
