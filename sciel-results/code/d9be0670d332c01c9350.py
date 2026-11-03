"""Combined: block-level Strassen (inspiration structure) + verification & deadline guard (refinement)."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 4, 4, 6
    rank = 56

    U = [[0] * rank for _ in range(n * m)]   # 16 x 77
    V = [[0] * rank for _ in range(m * p)]   # 24 x 77
    W = [[0] * rank for _ in range(p * n)]   # 24 x 77

    # Outer Strassen: partition A(4x4) into 2x2 blocks of 2x2,
    # B(4x6) into 2x2 blocks of 2x3, C(4x6) into 2x2 blocks of 2x3.
    # Block ids: 0=00, 1=01, 2=10, 3=11.
    # Each step: (A_coeffs, B_coeffs, C_coeffs) as list of (block_id, coeff).
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])),
    ]

    # Global index mappers.
    # A block (2x2): local (r,s) -> global (I,J). Block id b: I=(b//2)*2+r, J=(b%2)*2+s
    def gA(b, r, s):
        return (b // 2) * 2 + r, (b % 2) * 2 + s

    # B block (2x3): local (s,k) -> global (J,K). Block id b: J=(b//2)*2+s, K=(b%2)*3+k
    def gB(b, s, k):
        return (b // 2) * 2 + s, (b % 2) * 3 + k

    # C block (2x3): local (r,k) -> global (I,K). Block id b: I=(b//2)*2+r, K=(b%2)*3+k
    def gC(b, r, k):
        return (b // 2) * 2 + r, (b % 2) * 3 + k

    # Inner decomposition for 2x2 * 2x3 -> 2x3.
    # Part A: Strassen for first 2 columns (2x2 * 2x2), 7 terms.
    #   local A(r,s), B(s,k) with r,s,k in {0,1}, C(r,k).
    inner_strassen = [
        (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])),
        (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])),
        (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])),
        (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])),
        (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1), (0, 0, -1)])),
        (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])),
        (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])),
    ]

    # Part B: single column (2x2 * 2x1 -> 2x1), rank 1.
    #   T[r,s,2] = 1 for all r,s. Rank-1: U[r]=1, V[s]=1, W[2,r]=1.
    inner_schoolbook = [
        (([(0, 0, 1), (1, 0, 1)], [(0, 2, 1), (1, 2, 1)], [(0, 2, 1), (1, 2, 1)])),
    ]

    def add_term(rank_idx, A_terms, B_terms, C_terms, A_outer, B_outer, C_outer):
        for (r, s, ca) in A_terms:
            for (b, cb) in A_outer:
                I, J = gA(b, r, s)
                U[I * m + J][rank_idx] += ca * cb
        for (s, k, cb) in B_terms:
            for (b, cb2) in B_outer:
                J, K = gB(b, s, k)
                V[J * p + K][rank_idx] += cb * cb2
        for (r, k, cc) in C_terms:
            for (b, cb2) in C_outer:
                I, K = gC(b, r, k)
                W[K * n + I][rank_idx] += cc * cb2

    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = i * 8
        for j, (Ai, Bi, Ci) in enumerate(inner_strassen):
            add_term(base + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
        for j, (Ai, Bi, Ci) in enumerate(inner_schoolbook):
            add_term(base + 7 + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)

        if time.time() > deadline - 20:
            break

    # Verification: rebuild tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # and compare to the standard matmul tensor T[i,j,k] = A[i,j]*B[j,k] = delta structure.
    # For matmul tensor, T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0? No.
    # The matmul tensor T_{ijk} = A_{ij} B_{jk} C^{ik} contraction: T[i,j,k] = 1 always? 
    # Actually the tensor for matmul is T[i,j,k] = 1 for all i,j,k (the structure tensor).
    # Verify: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal 1 for all i,j,k.
    import numpy as np
    if time.time() < deadline - 5:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)
        # Build T[i,j,k]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            # u indexed by i*m+j, v by j*p+k, w by k*n+i
            # outer product over (i,j), (j,k), (k,i) -> sum over shared indices
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        # The matmul structure tensor is all ones.
        ok = np.allclose(T, 1.0)
        if not ok:
            # Fallback: schoolbook (rank 96) is guaranteed correct.
            rank2 = n * m * p
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i*m+j][r] = 1
                        V2[j*p+k][r] = 1
                        W2[k*n+i][r] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
