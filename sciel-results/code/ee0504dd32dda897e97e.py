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

    U, V, W = get_rank_77()
    
    # Convert to numpy for verification
    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    
    # Verify
    T = np.zeros((n, m, p))
    for r in range(77):
        u = Um[:, r]
        v = Vm[:, r]
        w = Wm[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    if not np.allclose(T, 1.0, atol=1e-10):
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
    
    # Try to reduce rank by zeroing out columns that are linearly dependent or zero
    # Check if any rank dimension has all zeros
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Find columns that are all zero
    zero_cols = []
    for r in range(77):
        if np.all(U_arr[:, r] == 0) or np.all(V_arr[:, r] == 0) or np.all(W_arr[:, r] == 0):
            zero_cols.append(r)
    
    if zero_cols:
        # Remove zero columns
        keep = [r for r in range(77) if r not in zero_cols]
        new_rank = len(keep)
        U_new = [[U[i][r] for r in keep] for i in range(n*m)]
        V_new = [[V[i][r] for r in keep] for i in range(m*p)]
        W_new = [[W[i][r] for r in keep] for i in range(p*n)]
        
        # Verify new solution
        Um_new = np.array(U_new, dtype=float)
        Vm_new = np.array(V_new, dtype=float)
        Wm_new = np.array(W_new, dtype=float)
        T_new = np.zeros((n, m, p))
        for r in range(new_rank):
            u = Um_new[:, r]
            v = Vm_new[:, r]
            w = Wm_new[:, r]
            T_new += np.einsum('i,j,k->ijk', u, v, w)
        
        if np.allclose(T_new, 1.0, atol=1e-10):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try local search: perturb entries slightly and see if we can reduce rank
    # This is a heuristic - try to find a lower-rank decomposition
    
    # Given time constraints, return the best we have
    return {"U": U, "V": V, "W": W}
