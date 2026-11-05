import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        """
        Compute Khatri-Rao product of A (M x R) and B (N x R).
        Result is (M*N) x R.
        """
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=200, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            # Solve for U
            KR_VW = khatri_rao(V, W)
            try:
                U_new, _, _, _ = np.linalg.lstsq(KR_VW.T, T1.reshape(T1.shape[0], -1).T, rcond=None)
                U_new = U_new.T
            except:
                break
            
            # Solve for V
            KR_WU = khatri_rao(W, U)
            try:
                V_new, _, _, _ = np.linalg.lstsq(KR_WU.T, T2.reshape(T2.shape[0], -1).T, rcond=None)
                V_new = V_new.T
            except:
                break
                
            # Solve for W
            KR_UV = khatri_rao(U, V)
            try:
                W_new, _, _, _ = np.linalg.lstsq(KR_UV.T, T3.reshape(T3.shape[0], -1).T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            # Check convergence
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        return U, V, W

    def snap_to_half_integer(U, V, W, T):
        """Snap entries to nearest half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        if np.allclose(recon, T, atol=1e-9):
            return U_s, V_s, W_s
        return None

    # New Strategy: Drop and Refit
    # Try to reduce rank by 1, 2, ...
    
    rng = np.random.default_rng(42)
    
    for target_rank in range(49, 44, -1):
        if time.time() > deadline - 20:
            break
            
        drop_count = 50 - target_rank
        found = False
        
        num_attempts = 10 if drop_count == 1 else 5
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
                
            # Select columns to keep
            if drop_count == 1:
                drop_idx = rng.integers(0, 50)
                keep_idx = [i for i in range(50) if i != drop_idx]
            elif drop_count == 2:
                drop_set = set(rng.choice(50, size=2, replace=False))
                keep_idx = [i for i in range(50) if i not in drop_set]
            else:
                continue
                
            U_init = U50[:, keep_idx].copy()
            V_init = V50[:, keep_idx].copy()
            W_init = W50[:, keep_idx].copy()
            
            # Refit using ALS
            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=80)
            
            # Try to snap to half-integers
            snapped = snap_to_half_integer(U_sol, V_sol, W_sol, T)
            if snapped is not None:
                U_sol, V_sol, W_sol = snapped
            
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
