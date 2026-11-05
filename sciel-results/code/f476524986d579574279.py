import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# Precompute the target tensor
_T = np.zeros((8, 28, 14))
for _i in range(2):
    for _j in range(4):
        for _k in range(7):
            _T[_i * 4 + _j, _j * 7 + _k, _k * 2 + _i] = 1.0
_T1 = _T.reshape(8, -1)
_T2 = _T.reshape(28, -1)
_T3 = _T.reshape(14, -1)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    T = _T
    T1, T2, T3 = _T1, _T2, _T3

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

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
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M2
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            # M3
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M4
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # M5
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            # M6
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M7
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

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

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(U_init, V_init, W_init, max_iters=150, tol=1e-14):
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            try:
                KR_VW = khatri_rao(V, W)
                U_new, *_ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break
            try:
                KR_WU = khatri_rao(W, U)
                V_new, *_ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break
            try:
                KR_UV = khatri_rao(U, V)
                W_new, *_ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        
        return U, V, W

    def polish_decomposition(U, V, W, max_iters=200):
        """High-precision polish pass with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            try:
                KR_VW = khatri_rao(V, W)
                U_new, *_ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break
            try:
                KR_WU = khatri_rao(W, U)
                V_new, *_ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break
            try:
                KR_UV = khatri_rao(U, V)
                W_new, *_ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < 1e-16:
                break
        
        return U, V, W

    def try_remove_column(U, V, W, col_idx, max_iters=50):
        """
        Try to remove column col_idx by re-solving the remaining columns.
        Returns (U_new, V_new, W_new, success)
        """
        R = U.shape[1]
        if R <= 1:
            return None, None, None, False
        
        # Remove the column
        keep = [i for i in range(R) if i != col_idx]
        U_init = U[:, keep].copy()
        V_init = V[:, keep].copy()
        W_init = W[:, keep].copy()
        
        # Re-solve with the remaining columns
        U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iters=max_iters)
        
        recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
        residual = np.linalg.norm(recon - T)
        
        if residual < 1e-8:
            # Polish
            U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol, max_iters=100)
            if verify_decomp(U_sol, V_sol, W_sol, atol=1e-8):
                return U_sol, V_sol, W_sol, True
        
        return None, None, None, False

    # Strategy: Exploit n=2 by trying to remove columns greedily
    # Start from rank 50 and try to reduce
    
    current_U, current_V, current_W = U50.copy(), V50.copy(), W50.copy()
    current_rank = 50
    
    # Try removing columns one by one
    for target_rank in range(49, 41, -1):
        if time.time() > deadline - 20:
            break
        
        if current_rank != target_rank:
            # We need to reduce from current_rank to target_rank
            # Try removing (current_rank - target_rank) columns
            cols_to_remove = current_rank - target_rank
            removed = 0
            attempts = 0
            max_attempts = 200
            
            while removed < cols_to_remove and attempts < max_attempts:
                if time.time() > deadline - 10:
                    break
                
                attempts += 1
                # Try each column
                for col_idx in range(current_U.shape[1]):
                    if time.time() > deadline - 10:
                        break
                    
                    U_new, V_new, W_new, success = try_remove_column(
                        current_U, current_V, current_W, col_idx, max_iters=40
                    )
                    
                    if success:
                        current_U, current_V, current_W = U_new, V_new, W_new
                        current_rank -= 1
                        removed += 1
                        break
                
                if removed < cols_to_remove and attempts >= max_attempts:
                    break
            
            if current_rank == target_rank:
                best_rank = target_rank
                best_U, best_V, best_W = current_U, current_V, current_W
                break
            else:
                # Reset to best found so far
                current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
                current_rank = best_rank
    
    # Also try random restarts with ALS for lower ranks
    rng = np.random.default_rng(42)
    
    for target_rank in range(best_rank - 1, max(40, best_rank - 8), -1):
        if time.time() > deadline - 15:
            break
        
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart < 5:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif restart < 10:
                # Perturb the best solution
                U_init = best_U[:, :target_rank].copy()
                V_init = best_V[:, :target_rank].copy()
                W_init = best_W[:, :target_rank].copy()
                U_init += rng.normal(0, 0.1, U_init.shape)
                V_init += rng.normal(0, 0.1, V_init.shape)
                W_init += rng.normal(0, 0.1, W_init.shape)
            else:
                # Try different subsets
                idx = rng.choice(best_rank, target_rank, replace=False)
                idx = np.sort(idx)
                U_init = best_U[:, idx].copy()
                V_init = best_V[:, idx].copy()
                W_init = best_W[:, idx].copy()
            
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iters=100)
            
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-6:
                # High-precision polish
                U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol, max_iters=150)
                
                if verify_decomp(U_sol, V_sol, W_sol, atol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
