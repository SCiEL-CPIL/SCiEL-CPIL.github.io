import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Build the rank-28 Strassen-based decomposition
    def build_strassen_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        # 2x2 * 2x2 Strassen blocks for j in {0,1}, k in {0,1} and k in {2,3}
        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        # Schoolbook for remaining
        # A[:,2] * B[2, 0:2]
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        # A[:,0:2] * B[0:2, 4]
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        # A[:,2] * B[2, 2:5]
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank, f"col={col} != {rank}"
        return U, V, W

    def check_decomposition(U, V, W, T):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < 1e-6

    # Build rank-28 and verify
    U28, V28, W28 = build_strassen_28()
    valid_28 = check_decomposition(U28, V28, W28, T)

    if not valid_28:
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try rank-1 peeling to reduce below 28
    # A rank-1 term has u (nm), v (mp), w (pn) vectors
    # We search for a rank-1 term such that T - u⊗v⊗w has a decomposition of rank 27
    # Strategy: enumerate candidate rank-1 terms consistent with the support of T
    # The support of T: for each (i,j,k) in [n]x[m]x[p], entry (i*m+j, j*p+k, k*n+i) is 1
    # A rank-1 term u⊗v⊗w contributes to (a,b,c) where u[a]v[b]w[c] != 0
    # For the residual to have lower rank, the rank-1 term must "cover" some of T's support

    # We'll try a structured search: pick a subset of the 30 support points and try to fit a rank-1 term
    # that exactly matches T on those points and is zero elsewhere in a way that helps.

    # Simpler approach: try all rank-1 terms where u, v, w are supported on small subsets
    # and have integer/half-integer values, then check if residual has rank 27.

    # For efficiency, we'll enumerate candidate rank-1 terms based on the structure.
    # Since T is sparse (30 non-zeros), a good rank-1 term should align with several of these.

    # We'll use a random + structured search with the time budget.
    best_rank = 28
    best_result = (U28, V28, W28)

    # Try to peel rank-1 terms
    # We attempt to find if rank 27 is achievable
    # A necessary condition: T must have a rank-1 summand
    # We search for u, v, w (with half-integer entries) such that
    # T - u⊗v⊗w can be decomposed with rank 27.

    # Given the complexity, we'll try a limited structured search.
    # The rank-1 term must have u, v, w with entries in {-1, -0.5, 0, 0.5, 1} typically.

    # We'll enumerate candidate rank-1 terms by selecting supports.
    # For each candidate, we check if the residual has a rank-27 decomposition.
    # To check rank 27, we can try to solve the linear system for the remaining 27 terms.

    # Given time constraints, we'll do a focused search.
    # The most likely rank-1 terms to peel are those that correspond to "diagonal" patterns.

    # Let's try: for each triple (i0, j0, k0), consider the rank-1 term that is 1 at
    # (i0*m+j0, j0*p+k0, k0*n+i0) and 0 elsewhere. This is the schoolbook term.
    # Peeling this would leave rank 29, not helpful.

    # Better: try rank-1 terms that cover multiple support points.
    # For example, a term where u has support on {i1*m+j, i2*m+j} for fixed j,
    # v has support on {j*p+k1, j*p+k2} for fixed j, w has support on {k1*n+i1, k2*n+i2}.

    # We'll do a random search with the time budget, trying to find a valid rank-27 decomposition
    # by initializing from the rank-28 decomposition and trying to merge two terms into one.

    # Actually, the cleanest approach: try to find a rank-27 decomposition directly using
    # alternating minimization with the rank-28 solution as initialization, but this is complex.

    # Given the time budget, we'll return the rank-28 result if we can't find better quickly.
    # The rank-28 is already an improvement over 30.

    # Let's do a quick random search for rank 27 with a time limit.
    if time_left() > 20:
        # Try a few random initializations for rank 27
        R = 27
        found = False
        for attempt in range(5):
            if time_left() < 15:
                break
            U = U28[:, :R].copy()
            V = V28[:, :R].copy()
            W = W28[:, :R].copy()
            # Add noise
            U += 0.1 * np.random.randn(nm, R)
            V += 0.1 * np.random.randn(mp, R)
            W += 0.1 * np.random.randn(pn, R)

            # Alternating minimization
            for it in range(30):
                if time_left() < 15:
                    break
                # Solve for W
                B = np.zeros((nm * mp, R))
                for r in range(R):
                    B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W_new = np.zeros((pn, R))
                for w in range(pn):
                    b_w = T[:, :, w].flatten()
                    x, _, _, _ = np.linalg.lstsq(B, b_w, rcond=None)
                    W_new[w, :] = x
                W = W_new

                # Solve for U
                C = np.zeros((mp * pn, R))
                for r in range(R):
                    C[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U_new = np.zeros((nm, R))
                for i in range(nm):
                    b_i = T[i, :, :].flatten()
                    x, _, _, _ = np.linalg.lstsq(C, b_i, rcond=None)
                    U_new[i, :] = x
                U = U_new

                # Solve for V
                D = np.zeros((nm * pn, R))
                for r in range(R):
                    D[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V_new = np.zeros((mp, R))
                for v in range(mp):
                    b_v = T[:, v, :].flatten()
                    x, _, _, _ = np.linalg.lstsq(D, b_v, rcond=None)
                    V_new[v, :] = x
                V = V_new

            # Check and snap
            if check_decomposition(U, V, W, T):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_decomposition(U_snap, V_snap, W_snap, T):
                    best_rank = 27
                    best_result = (U_snap, V_snap, W_snap)
                    found = True
                    break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
