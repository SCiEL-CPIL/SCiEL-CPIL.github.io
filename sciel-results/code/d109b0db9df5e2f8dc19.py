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
                    s = Fraction(0)
                    for r in range(rank):
                        u_val = U[a_idx][r]
                        v_val = V[b_idx][r]
                        w_val = W[c_idx][r]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            # Convert to Fraction for exact arithmetic
                            # If input is float, Fraction(val) might be imprecise if val is not exact binary.
                            # But we snapped to 0.5, 1.0, etc. which are exact in binary.
                            # However, it's safer to check if it's close to a simple fraction.
                            # For now, assume inputs are exact floats representing simple rationals.
                            s += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if s != 1:
                        return False
        return True

    # ---------- Fast Numerical Greedy Rank-1 Extraction ----------
    def fast_numerical_greedy(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        
        # Initialize Residual R as the target tensor T
        R = np.zeros((num_u, num_v, num_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i * m + j, j * p + k, k * n + i] = 1.0

        U_list = []
        V_list = []
        W_list = []
        
        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            
            # Find the entry with max abs value
            idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            a_idx, b_idx, c_idx = idx
            val = R[a_idx, b_idx, c_idx]
            
            if np.abs(val) < 1e-10:
                break
                
            # Project slices
            u = R[:, b_idx, c_idx] / val
            v = R[a_idx, :, c_idx] / val
            w = R[a_idx, b_idx, :] / val
            
            # Update R
            R -= val * np.einsum('i,j,k->ijk', u, v, w)
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)

        if np.linalg.norm(R) > 1e-8:
            return None

        # Snap to rationals
        def snap_to_rational(arr, tol=1e-5):
            snapped = []
            for val in arr:
                best = None
                best_err = tol
                for den in [1, 2]:
                    for num in range(-10, 11):
                        candidate = num / den
                        err = abs(val - candidate)
                        if err < best_err:
                            best_err = err
                            best = candidate
                if best is None:
                    return None
                snapped.append(best)
            return snapped

        U_snapped = []
        V_snapped = []
        W_snapped = []
        
        for u in U_list:
            us = snap_to_rational(u)
            if us is None: return None
            U_snapped.append(us)
        for v in V_list:
            vs = snap_to_rational(v)
            if vs is None: return None
            V_snapped.append(vs)
        for w in W_list:
            ws = snap_to_rational(w)
            if ws is None: return None
            W_snapped.append(ws)
            
        rank = len(U_snapped)
        U_final = [[0.0] * rank for _ in range(num_u)]
        V_final = [[0.0] * rank for _ in range(num_v)]
        W_final = [[0.0] * rank for _ in range(num_w)]
        
        for r in range(rank):
            for i in range(num_u):
                U_final[i][r] = U_snapped[r][i]
            for j in range(num_v):
                V_final[j][r] = V_snapped[r][j]
            for k in range(num_w):
                W_final[k][r] = W_snapped[r][k]
                
        return {"U": U_final, "V": V_final, "W": W_final}

    best = strassen_block_solution()
    
    # Try to reduce rank to 35
    if time.time() < deadline - 20:
        res = fast_numerical_greedy(35, deadline - 20)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res

    return best
# EVOLVE-BLOCK-END
