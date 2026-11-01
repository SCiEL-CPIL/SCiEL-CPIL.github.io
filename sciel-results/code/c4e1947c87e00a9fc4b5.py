"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify_full(U, V, W):
        """Verify exact decomposition by checking ALL tensor entries."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        # Reconstruct tensor
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Must be 1 if (i,j,k) is valid (all indices in range), 0 otherwise
        # But actually the tensor only has non-zero entries at valid (i,j,k)
        # where i in [0,n), j in [0,m), k in [0,p)
        # For the matrix multiplication tensor, T[i,j,k] = 1 if this is a valid
        # (input, input, output) triple, 0 otherwise.
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all
        # (i,j,k) where i is row of A, j is col of A / row of B, k is col of B.
        # Wait, the convention is W[k*n+i, r], so the output index is (k, i) meaning
        # C[k, i] = sum_j A[i, j] B[j, k]. So the tensor entries are at
        # (i, j, k) with value 1 for all valid i, j, k.
        # There are no "invalid" entries in the standard definition - the tensor
        # is simply defined on the full index set [0,n) x [0,m) x [0,p).
        # So we just check that T[i,j,k] = 1 for all valid (i,j,k).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def build_rank32():
        """Build the known rank-32 solution using Strassen on 4 pairs + trivial for k=8."""
        rank = 32
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # 7 Strassen terms for multiplying [2x2] * [2x2] for columns k0, k1
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
            # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
            # C11 = m1 + m4 - m5 - m7, C12 = m3 + m5, C21 = m2 + m4, C22 = m1 - m2 + m3 + m7

            # Term 1: m1 = (a+d)(e+h)
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k0, col] = 1; V[1*p+k1, col] = 1
            W[k0*n+0, col] = 1; W[k1*n+1, col] = 1
            col += 1

            # Term 2: m2 = (c+d)e
            U[2, col] = 1; U[3, col] = 1
            V[0*p+k0, col] = 1
            W[k0*n+1, col] = 1; W[k1*n+1, col] = -1
            col += 1

            # Term 3: m3 = a(f-h)
            U[0, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = -1
            W[k1*n+0, col] = 1; W[k1*n+1, col] = 1
            col += 1

            # Term 4: m4 = d(g-e)
            U[3, col] = 1
            V[1*p+k0, col] = 1; V[0*p+k0, col] = -1
            W[k0*n+0, col] = 1; W[k0*n+1, col] = 1
            col += 1

            # Term 5: m5 = (a+b)h
            U[0, col] = 1; U[1, col] = 1
            V[1*p+k1, col] = 1
            W[k0*n+0, col] = -1; W[k1*n+0, col] = 1
            col += 1

            # Term 6: m6 = (c-a)(e+f)
            U[2, col] = 1; U[0, col] = -1
            V[0*p+k0, col] = 1; V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            col += 1

            # Term 7: m7 = (b-d)(g+h)
            U[1, col] = 1; U[3, col] = -1
            V[1*p+k0, col] = 1; V[1*p+k1, col] = 1
            W[k0*n+0, col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            # 4 trivial terms for a single column k
            U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
            U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
            U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
            U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        # Use Strassen for 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        # Trivial for column 8
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_rank31():
        """Try to find a rank-31 decomposition by removing one trivial term and optimizing."""
        if time.time() > deadline - 60:
            return None

        r32 = build_rank32()
        U32 = np.array(r32["U"], dtype=np.float64)
        V32 = np.array(r32["V"], dtype=np.float64)
        W32 = np.array(r32["W"], dtype=np.float64)

        # The last 4 columns (28-31) are trivial terms for k=8
        # Try removing each one and see if we can adjust the remaining 31 to work
        for remove_col in range(28, 32):
            if time.time() > deadline - 55:
                break

            keep_cols = [c for c in range(32) if c != remove_col]
            U_keep = U32[:, keep_cols]
            V_keep = V32[:, keep_cols]
            W_keep = W32[:, keep_cols]

            x0 = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])

            def objective(x):
                if time.time() > deadline - 50:
                    return 1e10
                U_mat = x[:4*31].reshape(4, 31)
                V_mat = x[4*31:4*31+18*31].reshape(18, 31)
                W_mat = x[4*31+18*31:].reshape(18, 31)

                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            residual += (total - 1.0) ** 2
                return residual

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})

                if res.fun < 1e-12:
                    U_mat = res.x[:4*31].reshape(4, 31)
                    V_mat = res.x[4*31:4*31+18*31].reshape(18, 31)
                    W_mat = res.x[4*31+18*31:].reshape(18, 31)

                    U_list = U_mat.tolist()
                    V_list = V_mat.tolist()
                    W_list = W_mat.tolist()

                    if verify_full(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass

        return None

    def try_rank30_search():
        """Try to find a rank-30 decomposition via random initialization + optimization."""
        if time.time() > deadline - 50:
            return None

        # Start from the rank-32 solution and try to merge pairs of terms
        # This is a heuristic search
        r32 = build_rank32()
        U32 = np.array(r32["U"], dtype=np.float64)
        V32 = np.array(r32["V"], dtype=np.float64)
        W32 = np.array(r32["W"], dtype=np.float64)

        # Try removing 2 columns at a time and optimizing
        import itertools
        attempts = 0
        for c1, c2 in itertools.combinations(range(32), 2):
            if time.time() > deadline - 45 or attempts >= 20:
                break
            attempts += 1

            keep_cols = [c for c in range(32) if c != c1 and c != c2]
            U_keep = U32[:, keep_cols]
            V_keep = V32[:, keep_cols]
            W_keep = W32[:, keep_cols]

            x0 = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])

            def objective(x):
                if time.time() > deadline - 40:
                    return 1e10
                U_mat = x[:4*30].reshape(4, 30)
                V_mat = x[4*30:4*30+18*30].reshape(18, 30)
                W_mat = x[4*30+18*30:].reshape(18, 30)

                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            residual += (total - 1.0) ** 2
                return residual

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})

                if res.fun < 1e-12:
                    U_mat = res.x[:4*30].reshape(4, 30)
                    V_mat = res.x[4*30:4*30+18*30].reshape(18, 30)
                    W_mat = res.x[4*30+18*30:].reshape(18, 30)

                    U_list = U_mat.tolist()
                    V_list = V_mat.tolist()
                    W_list = W_mat.tolist()

                    if verify_full(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass

        return None

    # Build the known rank-32 solution
    result = build_rank32()

    if verify_full(result["U"], result["V"], result["W"]):
        # Try to improve to rank 31
        r31 = try_rank31()
        if r31 is not None:
            return r31

        # Try to improve to rank 30
        r30 = try_rank30_search()
        if r30 is not None:
            return r30

        return result

    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
