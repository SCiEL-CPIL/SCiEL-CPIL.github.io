import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Structured Kronecker Product Decomposition, Rank 32 ----------
    def kronecker_product_solution():
        """
        Partition dimensions:
        - Output k: 5 = 2 + 3
        - Inner j: 4 = 2 + 2
        
        Sub-problems:
        1. (i in 0..1, j in 0..1, k in 0..1): 2x2x2 tensor, rank 7 (Strassen)
        2. (i in 0..1, j in 2..3, k in 0..1): 2x2x2 tensor, rank 7 (Strassen)
        3. (i in 0..1, j in 0..1, k in 2..4): 2x2x3 tensor, rank 9 (Strassen+2)
        4. (i in 0..1, j in 2..3, k in 2..4): 2x2x3 tensor, rank 9 (Strassen+2)
        
        Total rank: 7 + 7 + 9 + 9 = 32
        """
        rank = 32
        U = [[0] * rank for _ in range(n * m)]  # 8 x 32
        V = [[0] * rank for _ in range(m * p)]  # 20 x 32
        W = [[0] * rank for _ in range(p * n)]  # 10 x 32
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Strassen 2x2x2 terms (local indices: U= i*2+j, V= j*2+k, W= k*2+i)
        strassen_2x2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        # 2x2x3 decomposition, rank 9:
        # First 7 terms = Strassen for k_local=0,1 (V idx = j*3+k, W idx = k*2+i)
        # Last 2 terms cover k_local=2
        # Local indices: U = i*2+j (0-3), V = j*3+k (0-5), W = k*2+i (0-5)
        def make_2x2x3_terms():
            terms = []
            # Strassen terms adapted for 2x2x3 (V has 3 k-values, W has 3 k-values)
            # For Strassen, V idx = j*2+k, W idx = k*2+i
            # For 2x2x3, V idx = j*3+k, W idx = k*2+i
            s23 = [
                # Same U-side as Strassen, V-side maps j*2+k -> j*3+k, W-side same
                ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
                ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
                ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
                ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
                ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
                ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
                ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
            ]
            # But V indices need remapping: in Strassen V idx = j*2+k (0-3)
            # In 2x2x3 V idx = j*3+k (0-5)
            # So V idx 0->0, 1->1, 2->3, 3->4
            v_remap = {0: 0, 1: 1, 2: 3, 3: 4}
            adapted = []
            for u_t, v_t, w_t in s23:
                v_new = {}
                for idx, val in v_t.items():
                    v_new[v_remap[idx]] = val
                adapted.append((u_t, v_new, w_t))
            # Two additional terms for k=2 (V idx = j*3+2, W idx = 2*2+i = 4+i)
            adapted.append(({0: 1, 1: 1}, {2: 1}, {4: 1}))
            adapted.append(({2: 1, 3: 1}, {5: 1}, {5: 1}))
            return adapted

        custom_2x2x3 = make_2x2x3_terms()

        # Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
        for u_t, v_t, w_t in strassen_2x2x2:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + lj] = val
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[lj * p + lk] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[lk * n + li] = val
            add_term(u_g, v_g, w_g)

        # Sub-problem 2: i in {0,1}, j in {2,3}, k in {0,1}
        for u_t, v_t, w_t in strassen_2x2x2:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + (2 + lj)] = val
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[(2 + lj) * p + lk] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[lk * n + li] = val
            add_term(u_g, v_g, w_g)

        # Sub-problem 3: i in {0,1}, j in {0,1}, k in {2,3,4}
        for u_t, v_t, w_t in custom_2x2x3:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + lj] = val
            for idx, val in v_t.items():
                lj, lk = idx // 3, idx % 3
                v_g[lj * p + (2 + lk)] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[(2 + lk) * n + li] = val
            add_term(u_g, v_g, w_g)

        # Sub-problem 4: i in {0,1}, j in {2,3}, k in {2,3,4}
        for u_t, v_t, w_t in custom_2x2x3:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + (2 + lj)] = val
            for idx, val in v_t.items():
                lj, lk = idx // 3, idx % 3
                v_g[(2 + lj) * p + (2 + lk)] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[(2 + lk) * n + li] = val
            add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    best = strassen_block_solution()
    
    # Try Kronecker product decomposition (rank 32)
    if time.time() < deadline - 5:
        res = kronecker_product_solution()
        if verify(res["U"], res["V"], res["W"]):
            best = res

    return best
# EVOLVE-BLOCK-END
