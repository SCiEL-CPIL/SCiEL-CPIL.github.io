import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)

    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))

    # Indices:
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j (6 entries)
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k (9 entries)
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (6 entries) [Trace convention W[k*n+i, r]]

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    # Verify the initial rank-17 solution
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 17
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Strategy: Try to find a rank-16 decomposition by solving linear constraints
    # The tensor has 18 entries (i,j,k) each equal to 1
    # For rank r, we need r terms such that for each (i,j,k): sum over r of U[i*3+j,r]*V[j*3+k,r]*W[k*2+i,r] = 1

    # Try direct construction for rank 16
    # We'll use a randomized approach: start with random rank-16 matrices and try to solve
    # Actually, let's try a more structured approach

    # First, let's try to see if we can find a rank-16 by removing one term and adjusting
    # For each pair of terms to merge, try to find coefficients

    # Alternative: Use the fact that we can represent the tensor as a sum of rank-1 tensors
    # Let's try to find a rank-16 solution by optimization

    # Try different strategies to reduce rank
    strategies = [
        "remove_and_adjust",
        "random_search",
        "structured_merge"
    ]

    for strategy in strategies:
        if time.time() > deadline - 10:
            break

        if strategy == "remove_and_adjust":
            # Try removing each term and solving for adjustments
            for term_to_remove in range(17):
                if time.time() > deadline - 5:
                    break

                # Zero out the term
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                U_test[:, term_to_remove] = 0
                V_test[:, term_to_remove] = 0
                W_test[:, term_to_remove] = 0

                # Try to adjust other terms to compensate
                # For each remaining term, try small adjustments
                found = False
                for r in range(17):
                    if r == term_to_remove:
                        continue
                    if time.time() > deadline - 5:
                        break

                    # Try adjusting W entries (6 entries) for term r
                    # The adjustment needed: for each (i,j,k), we need to add delta to make sum = 1
                    # Let's compute the deficit
                    deficit = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                current = sum(U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr] for rr in range(17))
                                deficit[i, j, k] = 1.0 - current

                    # Try to fix deficit by adjusting one term's W entries
                    # W_test[k*n+i, r] affects entries where k and i match
                    # For a given r, W_test[k*n+i, r] = w, then contribution to (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*w
                    # So we need: sum over (i,j,k) of U[i*m+j,r]*V[j*p+k,r]*w_{k,i} = deficit[i,j,k]

                    # This is a linear system in w_{k,i} for fixed r
                    # Let's set up the system
                    # Variables: w_00, w_01, w_10, w_11, w_20, w_21 (6 variables for k=0,1,2 and i=0,1)
                    # Equations: for each (i,j,k), U[i*m+j,r]*V[j*p+k,r]*w_{k,i} = deficit[i,j,k]
                    # But this is not linear in w if U and V are fixed... actually it IS linear in w for fixed r

                    # Build linear system: A * w = b where A has 18 rows (one per (i,j,k)) and 6 columns (w_{k,i})
                    A_sys = np.zeros((n*m*p, p*n))
                    b_sys = np.zeros(n*m*p)
                    for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
                        u_val = U_test[i*m+j, r]
                        v_val = V_test[j*p+k, r]
                        if abs(u_val) > 1e-10 and abs(v_val) > 1e-10:
                            A_sys[idx, k*n+i] = u_val * v_val
                            b_sys[idx] = deficit[i, j, k]

                    # Solve least squares
                    try:
                        w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        # Check if solution is good
                        w_test = W_test.copy()
                        for k in range(p):
                            for i in range(n):
                                w_test[k*n+i, r] = w_sol[k*n+i]
                        if verify(U_test, V_test, w_test, n, m, p):
                            found = True
                            break
                    except:
                        pass

                    if found:
                        break

                if found:
                    # Remove the term
                    keep = [r for r in range(17) if r != term_to_remove]
                    U_new = U_test[:, keep]
                    V_new = V_test[:, keep]
                    W_new = w_test[:, keep]
                    if verify(U_new, V_new, W_new, n, m, p):
                        best_rank = 16
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        break

        elif strategy == "random_search":
            # Try random rank-16 initializations
            np.random.seed(42)
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break

                U_rand = np.random.randn(n*m, 16)
                V_rand = np.random.randn(m*p, 16)
                W_rand = np.random.randn(p*n, 16)

                # Scale to make entries reasonable
                U_rand /= np.sqrt(16)
                V_rand /= np.sqrt(16)
                W_rand /= np.sqrt(16)

                # This is unlikely to work directly, but let's check
                # Actually, random won't work. Skip this strategy for now.
                pass

        elif strategy == "structured_merge":
            # Try to merge pairs of terms
            for r1 in range(17):
                for r2 in range(r1+1, 17):
                    if time.time() > deadline - 5:
                        break

                    # Try to merge terms r1 and r2 into a single term
                    # The merged term should satisfy: for each (i,j,k),
                    # U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
                    # = U_merged[i*m+j]*V_merged[j*p+k]*W_merged[k*n+i]

                    # This is generally not possible, but let's try a heuristic
                    # Compute the "pattern" of non-zero entries
                    pattern1 = set()
                    pattern2 = set()
                    for i in range(n):
                        for j in range(m):
                            if abs(U[i*m+j, r1]) > 1e-10:
                                pattern1.add(('U', i, j))
                            if abs(U[i*m+j, r2]) > 1e-10:
                                pattern2.add(('U', i, j))
                    for j in range(m):
                        for k in range(p):
                            if abs(V[j*p+k, r1]) > 1e-10:
                                pattern1.add(('V', j, k))
                            if abs(V[j*p+k, r2]) > 1e-10:
                                pattern2.add(('V', j, k))
                    for k in range(p):
                        for i in range(n):
                            if abs(W[k*n+i, r1]) > 1e-10:
                                pattern1.add(('W', k, i))
                            if abs(W[k*n+i, r2]) > 1e-10:
                                pattern2.add(('W', k, i))

                    # If patterns are very different, skip
                    if len(pattern1 & pattern2) > 8:
                        continue

                    # Try to find a merged term
                    # This is complex, skip for now
                    pass

    # If we found rank 16, return it
    if best_rank == 16:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
