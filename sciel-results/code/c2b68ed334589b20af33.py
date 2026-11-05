import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Laderman's 23-term algorithm for 3x3*3x3
    # Each term is (u, v, w) where:
    #   u has 9 entries indexed by (i, j) for A[i,j]
    #   v has 9 entries indexed by (j, k) for B[j,k]
    #   w has 9 entries indexed by (i, k) for C[i,k]  -- note: standard Laderman uses C[i,k]
    # BUT the problem uses W[k*n+i] convention, so we need w indexed by (k, i)
    # Let's use the standard Laderman terms and adapt the W indexing.
    
    # Standard Laderman's algorithm terms (u, v, w) with w indexed by C[i,k] = i*3+k
    # We'll convert to W[k*n+i] indexing when building the full decomposition.
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,1,0]),
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,1]),
        ([0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,1,0, 0,0,1]),
    ]

    def verify_laderman():
        """Verify Laderman terms correctly decompose 3x3*3x3 with W[k*n+i] convention."""
        n3, m3, p3 = 3, 3, 3
        # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        T_target = np.zeros((n3*m3, m3*p3, p3*n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T_target[i*m3+j, j*p3+k, k*n3+i] = 1.0

        # Reconstruct from Laderman terms
        # u is indexed by (i,j) -> i*3+j
        # v is indexed by (j,k) -> j*3+k
        # w is indexed by (i,k) in standard Laderman -> i*3+k
        # But we need W[k*n+i], so we need to remap w from (i,k) indexing to (k,i) indexing
        T_comp = np.zeros((n3*m3, m3*p3, p3*n3))
        for u, v, w in laderman_terms:
            u_np = np.array(u, dtype=float)
            v_np = np.array(v, dtype=float)
            # Remap w from (i,k) -> i*3+k to (k,i) -> k*3+i
            w_np = np.array(w, dtype=float).reshape(3, 3).T.flatten()
            # Now w_np[k*3+i] corresponds to W[k*n+i]
            T_comp += np.einsum('a,b,c->abc', u_np, v_np, w_np)

        return np.allclose(T_comp, T_target, atol=1e-9)

    if time.time() < deadline - 10:
        if verify_laderman():
            R = 46
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]

            # For each block of 3 k-values, use Laderman decomposition
            for block in range(2):
                k_offset = block * 3
                for r, (u, v, w) in enumerate(laderman_terms):
                    idx = block * 23 + r
                    u_np = np.array(u, dtype=float)
                    v_np = np.array(v, dtype=float)
                    # Remap w from (i,k_local) -> i*3+k_local to (k_local,i) -> k_local*3+i
                    w_np = np.array(w, dtype=float).reshape(3, 3).T.flatten()

                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][idx] = u_np[i*m+j]

                    for j in range(m):
                        for k_local in range(3):
                            k = k_offset + k_local
                            V[j*p+k][idx] = v_np[j*3+k_local]

                    for k_local in range(3):
                        k = k_offset + k_local
                        for i in range(n):
                            W[k*n+i][idx] = w_np[k_local*3+i]

            # Final verification of the full 3x3x6 decomposition
            T_target_full = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target_full[i*m+j, j*p+k, k*n+i] = 1.0

            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            T_comp_full = np.zeros((n*m, m*p, p*n))
            for c in range(R):
                T_comp_full += np.einsum('a,b,c->abc', U_np[:,c], V_np[:,c], W_np[:,c])

            if np.allclose(T_comp_full, T_target_full, atol=1e-9):
                return {"U": U, "V": V, "W": W}

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
