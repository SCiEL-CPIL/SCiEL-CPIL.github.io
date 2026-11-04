"""6x6x6 matrix multiplication rank minimization.

Strategy (MODEL-AND-OPTIMIZE family):
  1. Build a guaranteed-correct baseline: Strassen tiling (rank 189).
  2. Build a Laderman-style 3x3 rank-23 decomposition from verified 2x2
     Strassen patterns plus derived patterns. Verify EACH pattern against
     random 3x3 matrices; only keep patterns that pass.
  3. Tile the verified 3x3 decomposition 2x2 for 6x6 (rank 184 if all 23
     patterns verify, else fall back to the partial set / Strassen).
  4. Verify the full 6x6 decomposition before returning.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock limit

    def time_ok(margin=5.0):
        return time.time() < deadline - margin

    def verify_decomposition(U, V, W, A, B, tol=1e-9):
        """Verify C = A@B is reproduced by the tensor decomposition."""
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        rank = U.shape[1]
        for r in range(rank):
            M_u = np.dot(U[:, r], A.flatten())
            M_v = np.dot(V[:, r], B.flatten())
            M_r = M_u * M_v
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += M_r * W[k * n + i, r]
        return np.max(np.abs(C_computed - C_expected)) < tol

    # ------------------------------------------------------------------
    # 2x2 Strassen patterns (guaranteed correct).
    # Pattern: (A_coeffs, B_coeffs, C_coeffs)
    #   A: (i, j) row i col j of A
    #   B: (j, k) row j col k of B
    #   C: (i, k) row i col k of C
    # ------------------------------------------------------------------
    def strassen_2x2_patterns():
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]

    def verify_3x3_pattern(a_c, b_c, c_c, A, B, C):
        """Check that the single-term contribution equals sum_j A[i,j]B[j,k]."""
        u = 0.0
        for (i, j), v in a_c.items():
            u += v * A[i, j]
        v = 0.0
        for (j, k), w in b_c.items():
            v += w * B[j, k]
        prod = u * v
        for (i, k), w in c_c.items():
            if abs(prod * w - C[i, k]) > 1e-9:
                return False
        return True

    def build_laderman_3x3():
        """
        Build a 3x3 rank-23 decomposition.
        Returns (U, V, W) each of shape (9, 23), or None if it can't be built.
        U indexed by i*3+j, V by j*3+k, W by k*3+i.
        """
        # Start from the 7 verified Strassen patterns (mapped into 3x3 with
        # the (0,0) 2x2 sub-block).
        patterns = []
        for a_c, b_c, c_c in strassen_2x2_patterns():
            patterns.append((dict(a_c), dict(b_c), dict(c_c)))

        # 12 additional patterns to reach rank 23.
        # These are derived candidates; each is verified before use.
        extra = [
            ({(0,0):1, (1,0):-1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
            ({(1,0):1, (2,0):-1}, {(1,1):1, (2,1):-1}, {(1,1):1, (2,1):-1}),
            ({(0,1):1, (1,1):-1}, {(0,2):1, (1,2):-1}, {(0,2):1, (1,2):-1}),
            ({(1,1):1, (2,1):-1}, {(1,2):1, (2,2):-1}, {(1,2):1, (2,2):-1}),
            ({(0,2):1, (1,2):-1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):-1}),
            ({(1,2):1, (2,2):-1}, {(1,0):1, (2,0):-1}, {(1,0):1, (2,0):-1}),
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}),
            ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
            ({(0,1):1, (1,2):1}, {(1,0):1, (2,0):1}, {(1,0):1, (2,0):1}),
            ({(1,2):1, (2,0):1}, {(2,1):1, (0,1):1}, {(2,1):1, (0,1):1}),
            ({(2,0):1, (0,1):1}, {(0,2):1, (1,2):1}, {(0,2):1, (1,2):1}),
        ]
        patterns.extend(extra)

        # Verify each pattern against several random 3x3 matrices.
        rng = np.random.RandomState(12345)
        verified = []
        for a_c, b_c, c_c in patterns:
            ok = True
            for _ in range(8):
                A = rng.randn(3, 3)
                B = rng.randn(3, 3)
                C = A @ B
                if not verify_3x3_pattern(a_c, b_c, c_c, A, B, C):
                    ok = False
                    break
            if ok:
                verified.append((a_c, b_c, c_c))

        # We need the full set to be a valid decomposition. Check that the
        # sum of all verified patterns reproduces C on random inputs.
        if len(verified) == 0:
            return None

        rank = len(verified)
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        for idx, (a_c, b_c, c_c) in enumerate(verified):
            for (i, j), val in a_c.items():
                U[i * 3 + j, idx] = val
            for (j, k), val in b_c.items():
                V[j * 3 + k, idx] = val
            for (i, k), val in c_c.items():
                W[k * 3 + i, idx] = val

        # Full-decomposition verification on random 3x3 inputs.
        for _ in range(20):
            A = rng.randn(3, 3)
            B = rng.randn(3, 3)
            C_expected = A @ B
            C_computed = np.zeros((3, 3))
            for r in range(rank):
                M_u = np.dot(U[:, r], A.flatten())
                M_v = np.dot(V[:, r], B.flatten())
                M_r = M_u * M_v
                for i in range(3):
                    for k in range(3):
                        C_computed[i, k] += M_r * W[k * 3 + i, r]
            if np.max(np.abs(C_computed - C_expected)) > 1e-8:
                # Not a valid full decomposition with this pattern set.
                return None

        return U, V, W

    def build_strassen_6x6():
        """Baseline: Strassen tiling, rank 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        strassen_patterns = strassen_2x2_patterns()
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return U, V, W

    def build_tiled_6x6(local_U, local_V, local_W):
        """Tile a 3x3 decomposition (9xR) into 6x6 (36 x 4R)."""
        block_size = 3
        blocks_per_dim = 2
        R = local_U.shape[1]
        rank = (blocks_per_dim ** 3) * R
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for r_local in range(R):
                        for i_local in range(3):
                            for j_local in range(3):
                                val = local_U[i_local * 3 + j_local, r_local]
                                if val != 0:
                                    i_global = I * block_size + i_local
                                    j_global = J * block_size + j_local
                                    U[i_global * m + j_global, col_idx] = val
                        for i_local in range(3):
                            for j_local in range(3):
                                val = local_V[i_local * 3 + j_local, r_local]
                                if val != 0:
                                    i_global = J * block_size + i_local
                                    j_global = K * block_size + j_local
                                    V[i_global * p + j_global, col_idx] = val
                        for k_local in range(3):
                            for i_local in range(3):
                                val = local_W[k_local * 3 + i_local, r_local]
                                if val != 0:
                                    k_global = K * block_size + k_local
                                    i_global = I * block_size + i_local
                                    W[k_global * n + i_global, col_idx] = val
                        col_idx += 1
        return U, V, W

    # ------------------------------------------------------------------
    # Build baseline first (guaranteed valid).
    # ------------------------------------------------------------------
    U_base, V_base, W_base = build_strassen_6x6()
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    base_valid = verify_decomposition(U_base, V_base, W_base, A_test, B_test)

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 189

    # Try the Laderman 3x3 -> tiled 6x6 path.
    if time_ok(30):
        local = build_laderman_3x3()
        if local is not None:
            local_U, local_V, local_W = local
            R = local_U.shape[1]
            if R == 23:
                U_l, V_l, W_l = build_tiled_6x6(local_U, local_V, local_W)
                if verify_decomposition(U_l, V_l, W_l, A_test, B_test):
                    best_U, best_V, best_W = U_l, V_l, W_l
                    best_rank = 184
                    print("Laderman 6x6 valid, rank 184")
                else:
                    print("Laderman 6x6 full verification failed")

    # ------------------------------------------------------------------
    # Rank-reduction pass: try to remove linearly dependent terms from the
    # current best decomposition (a safe, monotonic refinement).
    # ------------------------------------------------------------------
    if time_ok(15):
        U_arr = best_U.astype(float)
        V_arr = best_V.astype(float)
        W_arr = best_W.astype(float)
        rank = U_arr.shape[1]
        # Build the "tensor" of the decomposition and check for redundant
        # terms by testing whether each term can be expressed as a linear
        # combination of the others while preserving the identity. This is
        # expensive; do a lightweight check: drop a term only if the
        # resulting decomposition still verifies.
        kept = list(range(rank))
        # Greedy: try to drop the term with the smallest norm contribution.
        norms = np.array([np.linalg.norm(U_arr[:, r]) *
                          np.linalg.norm(V_arr[:, r]) *
                          np.linalg.norm(W_arr[:, r]) for r in range(rank)])
        order = np.argsort(norms)
        new_kept = list(kept)
        for r in order:
            if r not in new_kept:
                continue
            candidate = [x for x in new_kept if x != r]
            if len(candidate) == 0:
                continue
            Uc = U_arr[:, candidate]
            Vc = V_arr[:, candidate]
            Wc = W_arr[:, candidate]
            if verify_decomposition(Uc, Vc, Wc, A_test, B_test):
                new_kept = candidate
                if not time_ok(10):
                    break
        if len(new_kept) < rank:
            best_U = U_arr[:, new_kept]
            best_V = V_arr[:, new_kept]
            best_W = W_arr[:, new_kept]
            best_rank = len(new_kept)
            print(f"Reduced rank to {best_rank}")

    # Final verification of the returned solution.
    if not verify_decomposition(best_U, best_V, best_W, A_test, B_test):
        # Fall back to the guaranteed baseline.
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 189

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
# EVOLVE-BLOCK-END
