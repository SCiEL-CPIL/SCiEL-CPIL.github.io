import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s kill

    # ---------- Rank-36 Strassen block decomposition ----------
    # Partition the 4-dimensional inner product into two 2-dimensional halves
    # and the 5-dimensional output into 2+3.
    # For the 2x2x2 blocks: use Strassen (rank 7)
    # For the 2x2x1 blocks: use standard (rank 4)
    # Total: 2 * 7 + 3 * 4 = 14 + 12 = 26... wait, let me recount.
    # Actually: 2 blocks of (2x2) * (2x2) -> output 2x2, using Strassen rank 7 each: 2*7 = 14
    # 3 blocks of (2x2) * (2x1) -> output 2x1, using standard rank 4 each: 3*4 = 12
    # Total = 26? No wait, the output dimension is 5, split as 2+3.
    # The 5 output rows: first 2 come from the 2x2x2 blocks, last 3 from 2x2x1 blocks.
    # For each of the 2 input column halves (q=0,1):
    #   - 2 output rows (k=0,1): 2x2x2 Strassen, rank 7
    #   - 3 output rows (k=2,3,4): 2x2x1 standard, rank 4 each
    # Total: 2 * (7 + 3*4) = 2 * 19 = 38? That's worse.
    # Let me re-examine the current program's approach.
    # Current: for q in range(2), s in range(3):
    #   if s < 2: Strassen 2x2x2 (7 terms)
    #   else: standard 2x2x1 (4 terms)
    # Total: 2 * (2*7 + 1*4) = 2 * 18 = 36. Yes!
    # The 5 output columns are split as 2+2+1? No, s ranges 0,1,2.
    # s=0,1: 2x2x2 Strassen blocks (7 each)
    # s=2: 2x2x1 standard block (4)
    # Per q: 7+7+4 = 18. Two q values: 36. Correct.

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

        # Strassen terms for 2x2x2: (u, v, w) where u,v,w are dicts over {0,1,2,3}
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        # Standard terms for 2x2x1:
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

    # ---------- Fast NumPy-based verification ----------
    def verify_numpy(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                for k in range(p):
                    b_idx = j * p + k
                    c_idx = k * n + i
                    prod = U_np[a_idx] * V_np[b_idx] * W_np[c_idx]
                    s = np.sum(prod)
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    # ---------- Exact rational verification ----------
    def verify_exact(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = Fraction(0)
                    for r in range(rank):
                        uv = U[a_idx][r] * V[b_idx][r]
                        if uv != 0:
                            s += uv * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Attempt to reduce rank via greedy elimination ----------
    # Try to find a rank-35 or lower solution by checking if any column
    # can be expressed as a linear combination of others, or by perturbing.
    # This is a heuristic; we time-box it.
    
    def try_greedy_reduction(U, V, W, max_time=30.0):
        """Try to reduce rank by eliminating linearly dependent columns."""
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Build the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check if removing column r still gives correct T
        # This is expensive, so we sample.
        
        # Alternative: check if the 3D array of (U[:,r], V[:,r], W[:,r]) 
        # has any column that's linearly dependent in a structured way.
        # For now, just verify the current solution and return it.
        # A more sophisticated approach would use rank-revealing decomposition
        # of the flattening of the tensor, but that's complex.
        
        return None  # No improvement found

    # ---------- Main ----------
    best = strassen_block_solution()
    
    # Verify
    if not verify_numpy(best["U"], best["V"], best["W"]):
        if not verify_exact(best["U"], best["V"], best["W"]):
            raise ValueError("Internal error: Strassen solution failed verification")
    
    # Try greedy reduction if time permits
    if time.time() < deadline - 35.0:
        improved = try_greedy_reduction(best["U"], best["V"], best["W"], 
                                         max_time=min(30.0, deadline - time.time() - 5.0))
        if improved is not None:
            best = improved
    
    return best
# EVOLVE-BLOCK-END
