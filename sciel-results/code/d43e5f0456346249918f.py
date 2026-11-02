"""Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
n, m, p = DIMENSIONS
N = n * m  # 20
M = m * p  # 25
P = p * n  # 20


def construct():
    """Construct a low-rank decomposition of the 4x5*5x5 matrix multiplication tensor."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit

    # The matrix multiplication tensor T has shape (N, M, P) where:
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k) triples
    # and T[a, b, c] = 0 otherwise
    #
    # We seek: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

    # Precompute the list of "on-diagonal" indices (where T=1)
    on_diag = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on_diag.append((a, b, c))

    # Precompute off-diagonal indices
    all_indices = list(product(range(N), range(M), range(P)))
    on_diag_set = set(on_diag)
    off_diag = [idx for idx in all_indices if idx not in on_diag_set]

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact within tolerance."""
        # Check on-diagonal entries
        for a, b, c in on_diag:
            val = float(np.dot(U[a], V[b] * W[c]))
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of off-diagonal entries for speed
        for a, b, c in off_diag[::max(1, len(off_diag) // 500)]:
            val = float(np.dot(U[a], V[b] * W[c]))
            if abs(val) > tol:
                return False
        return True

    def check_decomposition_full(U, V, W, tol=1e-6):
        """Full check of all entries."""
        for a, b, c in on_diag:
            val = float(np.dot(U[a], V[b] * W[c]))
            if abs(val - 1.0) > tol:
                return False
        for a, b, c in off_diag:
            val = float(np.dot(U[a], V[b] * W[c]))
            if abs(val) > tol:
                return False
        return True

    def alternating_minimize(U, V, W, r, max_iter=200):
        """
        Alternating minimization for the rank-r decomposition.
        For each factor, solve a least-squares problem with the other two fixed.
        """
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                return U, V, W, False

            # Solve for U given V, W
            for a in range(N):
                # For each (b,c), the contribution to T[a,b,c] is U[a,:] dot (V[b,:]*W[c,:])
                # We want sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For fixed a, this is a linear system in U[a,:]
                # A row for (b,c) is V[b,:]*W[c,:]
                # Build the system
                A = np.zeros((len(on_diag) + len(off_diag), r))
                rhs = np.zeros(len(on_diag) + len(off_diag))
                idx = 0
                for (a2, b, c) in on_diag:
                    if a2 == a:
                        A[idx] = V[b] * W[c]
                        rhs[idx] = 1.0
                    idx += 1
                for (a2, b, c) in off_diag:
                    if a2 == a:
                        A[idx] = V[b] * W[c]
                        rhs[idx] = 0.0
                    idx += 1
                # Only use rows where a2 == a
                mask_a = []
                for (a2, b, c) in on_diag:
                    if a2 == a:
                        mask_a.append((b, c, 1.0))
                for (a2, b, c) in off_diag:
                    if a2 == a:
                        mask_a.append((b, c, 0.0))
                if len(mask_a) > 0:
                    A_sub = np.array([V[b] * W[c] for b, c, _ in mask_a])
                    rhs_sub = np.array([t for _, _, t in mask_a])
                    U[a], _, _, _ = np.linalg.lstsq(A_sub, rhs_sub, rcond=None)

            # Solve for V given U, W
            for b in range(M):
                mask_b = []
                for (a, a2, c) in [(x[0], x[1], x[2]) for x in on_diag]:
                    if a2 == b:
                        mask_b.append((a, c, 1.0))
                for (a, a2, c) in [(x[0], x[1], x[2]) for x in off_diag]:
                    if a2 == b:
                        mask_b.append((a, c, 0.0))
                if len(mask_b) > 0:
                    A_sub = np.array([U[a] * W[c] for a, c, _ in mask_b])
                    rhs_sub = np.array([t for _, _, t in mask_b])
                    V[b], _, _, _ = np.linalg.lstsq(A_sub, rhs_sub, rcond=None)

            # Solve for W given U, V
            for c in range(P):
                mask_c = []
                for (a, b, c2) in [(x[0], x[1], x[2]) for x in on_diag]:
                    if c2 == c:
                        mask_c.append((a, b, 1.0))
                for (a, b, c2) in [(x[0], x[1], x[2]) for x in off_diag]:
                    if c2 == c:
                        mask_c.append((a, b, 0.0))
                if len(mask_c) > 0:
                    A_sub = np.array([U[a] * V[b] for a, b, _ in mask_c])
                    rhs_sub = np.array([t for _, _, t in mask_c])
                    W[c], _, _, _ = np.linalg.lstsq(A_sub, rhs_sub, rcond=None)

        # Final check
        if check_decomposition(U, V, W, tol=1e-5):
            return U, V, W, True
        return U, V, W, False

    # Try structured initializations for various ranks
    best_result = None

    # Strategy 1: Try to find low-rank solutions using structured patterns
    # The key insight: for (4,5,5), the lower bound is max(20,25,20) = 25
    # Let's try ranks from 25 upward with different initializations

    for r in range(25, min(101, 60)):  # Try up to rank 60
        if time.time() - start_time > deadline - 10:
            break

        found = False
        # Try multiple random initializations
        for seed in range(5):
            if time.time() - start_time > deadline - 5:
                break
            if found:
                break

            np_rng = np.random.default_rng(seed * 1000 + r)
            scale = 0.1
            U = np_rng.standard_normal((N, r)) * scale
            V = np_rng.standard_normal((M, r)) * scale
            W = np_rng.standard_normal((P, r)) * scale

            # Also try a structured initialization based on the "natural" grouping
            # c = j*p + k for some natural indexing
            if seed == 0:
                U = np.zeros((N, r))
                V = np.zeros((M, r))
                W = np.zeros((P, r))
                # Try to match the structure: for each (j,k), use one column
                # U[i*m+j, j*p+k] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, j*p+k] = 1
                # But this only works if r >= m*p = 25
                for j in range(m):
                    for k in range(p):
                        col = j * p + k
                        if col < r:
                            for i in range(n):
                                U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            for i in range(n):
                                W[k * n + i, col] = 1.0

            U, V, W, success = alternating_minimize(U, V, W, r, max_iter=100)

            if success:
                # Verify with full check
                if check_decomposition_full(U, V, W, tol=1e-5):
                    U_list = [[float(U[i, j]) for j in range(r)] for i in range(N)]
                    V_list = [[float(V[i, j]) for j in range(r)] for i in range(M)]
                    W_list = [[float(W[i, j]) for j in range(r)] for i in range(P)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break

        if found:
            break

    if best_result is not None:
        return best_result

    # Fallback: schoolbook decomposition (rank = 100)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(N)]
    V = [[0.0] * rank for _ in range(M)]
    W = [[0.0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
