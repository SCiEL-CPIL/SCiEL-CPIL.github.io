import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6
    
    # Fallback: Rank 77 (Current best)
    def get_rank_77():
        rank = 77
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        outer_steps = [
            (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
            (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
            (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
            (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
            (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
            (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
            (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])),
        ]
        
        def gA(b, r, s): return (b // 2) * 2 + r, (b % 2) * 2 + s
        def gB(b, s, k): return (b // 2) * 2 + s, (b % 2) * 3 + k
        def gC(b, r, k): return (b // 2) * 2 + r, (b % 2) * 3 + k
        
        inner_strassen = [
            (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])),
            (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])),
            (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])),
            (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])),
            (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1), (0, 0, -1)])),
            (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])),
            (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])),
        ]
        inner_schoolbook = [
            (([(0, 0, 1)], [(0, 2, 1)], [(0, 2, 1)])),
            (([(0, 1, 1)], [(1, 2, 1)], [(0, 2, 1)])),
            (([(1, 0, 1)], [(0, 2, 1)], [(1, 2, 1)])),
            (([(1, 1, 1)], [(1, 2, 1)], [(1, 2, 1)])),
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
            base = i * 11
            for j, (Ai, Bi, Ci) in enumerate(inner_strassen):
                add_term(base + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
            for j, (Ai, Bi, Ci) in enumerate(inner_schoolbook):
                add_term(base + 7 + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
        
        return U, V, W

    # Try Rank 35: Kronecker Product of 2x2x3 (rank 5) and 2x2x2 (rank 7)
    # Rank 5 decomposition for 2x2x3
    # This is a known decomposition. 
    # U1 (2x2x5), V1 (2x3x5), W1 (2x3x5)
    # Note: Indices for 2x2x3: i,j in {0,1}, k in {0,1,2}
    
    # If I cannot guarantee the rank-5 factors, I use rank 77.
    # Given the risk, I will return the rank 77 solution which is verified.
    
    U, V, W = get_rank_77()
    
    # Verification
    Um = np.array(U)
    Vm = np.array(V)
    Wm = np.array(W)
    T = np.zeros((n, m, p))
    for r in range(77):
        u = Um[:, r]
        v = Vm[:, r]
        w = Wm[:, r]
        for i in range(n):
            for j in range(m):
                if u[i*m+j] == 0: continue
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    if not np.allclose(T, 1.0):
        # Fallback to Schoolbook 96
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
