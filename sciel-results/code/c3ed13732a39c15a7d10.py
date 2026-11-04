import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

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
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    best_rank = 50

    # Strategy: Try to find rank reductions by checking if columns are linearly dependent
    # in the tensor space. We'll use a greedy approach with linear algebra.
    
    # For each target rank, try to find a valid decomposition
    for target_rank in range(49, 30, -1):
        if time.time() > deadline - 15:
            break
        
        success = False
        
        # Try multiple random subsets of columns from the rank-50 solution
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 13)
            
            if target_rank >= best_rank:
                idx = list(range(target_rank))
            else:
                idx = np.random.choice(best_rank, target_rank, replace=False)
            
            U_init = best_U[:, idx].copy()
            V_init = best_V[:, idx].copy()
            W_init = best_W[:, idx].copy()
            
            # Try to adjust the columns to exactly represent T
            # Use a simple iterative refinement
            U_try = U_init.copy()
            V_try = V_init.copy()
            W_try = W_init.copy()
            
            # Compute current reconstruction
            R = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
            residual = T - R
            
            # Try a few iterations of adjusting one factor at a time
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for c in range(target_rank):
                    vw = V_try[:, c] * W_try[:, c]
                    denom = np.sum(vw * vw)
                    if denom > 1e-12:
                        U_new = np.einsum('abk,bk->a', T, vw) / denom
                        # Check if this improves things
                        R_new = R - np.einsum('i,j,k->ijk', U_try[:, c], V_try[:, c], W_try[:, c]) + np.einsum('i,j,k->ijk', U_new, V_try[:, c], W_try[:, c])
                        if np.linalg.norm(T - R_new) < np.linalg.norm(T - R):
                            R = R_new
                            U_try[:, c] = U_new
                
                # Update V
                for c in range(target_rank):
                    uw = U_try[:, c] * W_try[:, c]
                    denom = np.sum(uw * uw)
                    if denom > 1e-12:
                        V_new = np.einsum('abk,ak->b', T, uw) / denom
                        R_new = R - np.einsum('i,j,k->ijk', U_try[:, c], V_try[:, c], W_try[:, c]) + np.einsum('i,j,k->ijk', U_try[:, c], V_new, W_try[:, c])
                        if np.linalg.norm(T - R_new) < np.linalg.norm(T - R):
                            R = R_new
                            V_try[:, c] = V_new
                
                # Update W
                for c in range(target_rank):
                    uv = U_try[:, c] * V_try[:, c]
                    denom = np.sum(uv * uv)
                    if denom > 1e-12:
                        W_new = np.einsum('abk,ab->k', T, uv) / denom
                        R_new = R - np.einsum('i,j,k->ijk', U_try[:, c], V_try[:, c], W_try[:, c]) + np.einsum('i,j,k->ijk', U_try[:, c], V_try[:, c], W_new)
                        if np.linalg.norm(T - R_new) < np.linalg.norm(T - R):
                            R = R_new
                            W_try[:, c] = W_new
                
                # Check convergence
                err = np.linalg.norm(T - R)
                if err < 1e-8:
                    break
            
            if np.linalg.norm(T - R) < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U_try * 2) / 2
                V_r = np.round(V_try * 2) / 2
                W_r = np.round(W_try * 2) / 2
                
                if verify_decomp(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    success = True
                    break
                else:
                    # Try integer rounding
                    U_i = np.round(U_try)
                    V_i = np.round(V_try)
                    W_i = np.round(W_try)
                    if verify_decomp(U_i, V_i, W_i):
                        best_U, best_V, best_W = U_i, V_i, W_i
                        best_rank = target_rank
                        success = True
                        break
        
        if not success:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
