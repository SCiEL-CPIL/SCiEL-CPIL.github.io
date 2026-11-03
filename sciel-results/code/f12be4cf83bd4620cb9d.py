import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = l_row * m + (a_col_start + l_col)
                U[global_idx, col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                V[global_idx, col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (c_col_start + l_col) * n + l_row
                W[global_idx, col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-8):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _als_decomposition(U_init, V_init, W_init, target_rank, max_iters=500, seed=42, deadline=None, tol=1e-12):
    """
    Alternating Least Squares for tensor decomposition.
    Given target tensor T (implicit as matmul tensor), find U,V,W of rank target_rank.
    Each step solves a linear least-squares problem for one factor.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    # Initialize from perturbed truncated version
    U = U_init[:, :target_rank].copy().astype(float)
    V = V_init[:, :target_rank].copy().astype(float)
    W = W_init[:, :target_rank].copy().astype(float)
    
    # Small random perturbation
    U += 0.05 * rng.randn(*U.shape)
    V += 0.05 * rng.randn(*V.shape)
    W += 0.05 * rng.randn(*W.shape)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # Compute current residual
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < tol:
            break
        
        # ALS step 1: Solve for U given V, W
        # For each (i,j): sum_r U[i,j,r] * V[j,*,r] * W[*,i,r] = T[i,j,:]
        # This is linear in U[i,j,:] for fixed j
        # Let's reshape: for fixed j, we need U[i,:,j] (n x rank) such that
        # sum_r U[i,j,r] * (V3[j,:,r] * W3[:,i,r]) = T[i,j,:]
        # Actually, let's think differently.
        
        # The reconstruction is: recon[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed (i,k), this is: recon[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # So for each (i,k), we have: T[i,j,k] = sum_r U[i,j,r] * M_ik[j,r] where M_ik[j,r] = V[j,k,r]*W[k,i,r]
        # This gives us for each (i,k): T[i,:,k] = U[i,:,:] @ M_ik  (n x rank @ rank x n -> n x n)
        # Wait, that's not quite right. Let me re-derive.
        
        # recon[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed i, k: recon[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # Define A_ik[j,r] = V[j,k,r] * W[k,i,r]
        # Then recon[i,j,k] = sum_r U[i,j,r] * A_ik[j,r] = (U[i] @ A_ik^T)[j]
        # where U[i] is the row of U corresponding to (i,j) for all j... 
        # Actually U is (n*m, rank), so U[i*m + j, :] is the vector for position (i,j).
        
        # Let's flatten: let u_idx = i*m + j
        # recon[i,j,k] = sum_r U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed (i,k): we have equations for all j:
        #   T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        # This is a linear system in U[i*m+j, :] for each j, but they're coupled through r.
        
        # Better approach: vectorize. 
        # For fixed i: we have m equations (one for each j) and p unknowns per j... 
        # Actually, let's just do a full least squares per factor.
        
        # For U: fix V and W. The model is linear in U.
        # Vectorize: let t = T.reshape(-1) (n*m*p elements)
        # Let u = U.reshape(-1) (n*m*rank elements)
        # The model: t = A_U @ u where A_U is (n*m*p) x (n*m*rank)
        # A_U[(i*m+j)*p+k, (i*m+j)*rank + r] = V[j*p+k, r] * W[k*n+i, r]
        
        # This is a huge matrix. Let's use the structure.
        # For each (i,j): t[i*m+j, :] = sum_r U[i*m+j, r] * (V[j, :, r] * W[:, i, r])
        # Let B_ij[r, k] = V[j*p+k, r] * W[k*n+i, r]  (rank x p)
        # Then t[i*m+j, :] = U[i*m+j, :] @ B_ij  (1 x rank @ rank x p -> 1 x p)
        
        # So for each (i,j): solve U[i*m+j, :] @ B_ij = T[i,j,:]
        # This is an underdetermined system (rank unknowns, p equations) when rank > p
        # We want the minimum-norm solution: U[i*m+j, :] = T[i,j,:] @ B_ij^T @ (B_ij @ B_ij^T)^{-1}
        # Or use lstsq.
        
        for i in range(n):
            for j in range(m):
                # Build B_ij: (rank x p)
                B_ij = np.zeros((target_rank, p))
                for r in range(target_rank):
                    for k in range(p):
                        B_ij[r, k] = V[j * p + k, r] * W[k * n + i, r]
                # Solve U[i*m+j, :] @ B_ij = T[i, j, :]
                # Equivalent to B_ij^T @ U[i*m+j, :]^T = T[i,j,:]^T
                rhs = T[i, j, :].copy()
                # Minimum norm solution
                try:
                    sol, residuals, rank_b, sv = lstsq(B_ij.T, rhs, rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # ALS step 2: Solve for V given U, W
        # recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed (j,k): T[:, j, k] = sum_r V[j*p+k, r] * (U[:, j, r] * W[k*n+?, r])
        # Hmm, this is trickier. Let me re-derive.
        # For fixed j: recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed (j, k): T[i, j, k] = sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r])
        # Let C_jk[r, i] = U[i*m+j, r] * W[k*n+i, r]  (rank x n)
        # Then T[:, j, k] = V[j*p+k, :] @ C_jk  (1 x rank @ rank x n -> 1 x n)
        
        for j in range(m):
            for k in range(p):
                C_jk = np.zeros((target_rank, n))
                for r in range(target_rank):
                    for i in range(n):
                        C_jk[r, i] = U[i * m + j, r] * W[k * n + i, r]
                rhs = T[:, j, k].copy()
                try:
                    sol, residuals, rank_c, sv = lstsq(C_jk.T, rhs, rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # ALS step 3: Solve for W given U, V
        # recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed (k,i): T[:, j, k]... wait, T is indexed (i,j,k)
        # For fixed (k, i): T[i, j, k] = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
        # Let D_ki[r, j] = U[i*m+j, r] * V[j*p+k, r]  (rank x m)
        # Then T[i, :, k] = W[k*n+i, :] @ D_ki  (1 x rank @ rank x m -> 1 x m)
        
        for k in range(p):
            for i in range(n):
                D_ki = np.zeros((target_rank, m))
                for r in range(target_rank):
                    for j in range(m):
                        D_ki[r, j] = U[i * m + j, r] * V[j * p + k, r]
                rhs = T[i, :, k].copy()
                try:
                    sol, residuals, rank_d, sv = lstsq(D_ki.T, rhs, rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
    
    return best_U, best_V, best_W, best_residual


def _gradient_optimize(U_init, V_init, W_init, target_rank, max_iters=3000, lr_init=0.01, seed=42, deadline=None):
    """Gradient-based optimization with adaptive learning rate and restarts."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rng = np.random.RandomState(seed)
    
    U = U_init[:, :target_rank].copy().astype(float)
    V = V_init[:, :target_rank].copy().astype(float)
    W = W_init[:, :target_rank].copy().astype(float)
    
    U += 0.1 * rng.randn(*U.shape)
    V += 0.1 * rng.randn(*V.shape)
    W += 0.1 * rng.randn(*W.shape)
    
    lr = lr_init
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-25:
            break
        
        dU3 = 2.0 * residual * np.einsum('jkr,kir->ijr', V3, W3)
        dV3 = 2.0 * residual * np.einsum('ijr,kir->jkr', U3, W3)
        dW3 = 2.0 * residual * np.einsum('ijr,jkr->kir', U3, V3)
        
        dU = dU3.reshape(n*m, -1)
        dV = dV3.reshape(m*p, -1)
        dW = dW3.reshape(p*n, -1)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        lr *= 0.9995
        
        if it > 0 and it % 300 == 0 and loss > best_residual * 2.0:
            U = best_U + 0.03 * rng.randn(*best_U.shape)
            V = best_V + 0.03 * rng.randn(*best_V.shape)
            W = best_W + 0.03 * rng.randn(*best_W.shape)
            lr = lr_init * 0.5
    
    return best_U, best_V, best_W, best_residual


def _polish(U, V, W, max_iters=1000, deadline=None, tol=1e-15):
    """High-precision polish pass using ALS with tight tolerance."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    rank = U.shape[1]
    
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = recon - T
        loss = np.sum(residual**2)
        
        if loss < best_residual:
            best_residual = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < tol:
            break
        
        # ALS for U
        for i in range(n):
            for j in range(m):
                B_ij = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        B_ij[r, k] = V[j * p + k, r] * W[k * n + i, r]
                rhs = T[i, j, :].copy()
                try:
                    sol, _, _, _ = lstsq(B_ij.T, rhs, rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # ALS for V
        for j in range(m):
            for k in range(p):
                C_jk = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        C_jk[r, i] = U[i * m + j, r] * W[k * n + i, r]
                rhs = T[:, j, k].copy()
                try:
                    sol, _, _, _ = lstsq(C_jk.T, rhs, rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # ALS for W
        for k in range(p):
            for i in range(n):
                D_ki = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        D_ki[r, j] = U[i * m + j, r] * V[j * p + k, r]
                rhs = T[i, :, k].copy()
                try:
                    sol, _, _, _ = lstsq(D_ki.T, rhs, rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
    
    return best_U, best_V, best_W, best_residual


def _try_rank_reduction(target_rank, U28, V28, W28, max_restarts=10, seed=0, deadline=None):
    """Try to find a rank-target_rank decomposition using ALS + gradient + polish."""
    n, m, p = DIMENSIONS
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Method 1: ALS
        U_r, V_r, W_r, residual = _als_decomposition(
            U28, V28, W28, target_rank,
            max_iters=300, seed=seed + restart, deadline=deadline
        )
        
        if residual < 1e-10:
            # Polish
            U_p, V_p, W_p, res_p = _polish(U_r, V_r, W_r, max_iters=500, deadline=deadline)
            if res_p < 1e-15:
                valid, _ = _verify_decomposition(U_p, V_p, W_p, tol=1e-10)
                if valid:
                    return U_p, V_p, W_p, target_rank
        
        # Method 2: Gradient
        U_g, V_g, W_g, res_g = _gradient_optimize(
            U28, V28, W28, target_rank,
            max_iters=2000, lr_init=0.005,
            seed=seed + restart + 100, deadline=deadline
        )
        
        if res_g < 1e-10:
            U_p, V_p, W_p, res_p = _polish(U_g, V_g, W_g, max_iters=500, deadline=deadline)
            if res_p < 1e-15:
                valid, _ = _verify_decomposition(U_p, V_p, W_p, tol=1e-10)
                if valid:
                    return U_p, V_p, W_p, target_rank
    
    return None, None, None, None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
    # Start with known rank-28 solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27
    if time.time() < deadline:
        for seed in range(20):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_rank_reduction(27, U28, V28, W28, max_restarts=1, seed=seed, deadline=deadline)
                if U_r is not None:
                    best_rank = 27
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
            except Exception:
                continue
    
    # If we found rank 27, try to go to rank 26
    if best_rank == 27 and time.time() < deadline:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        for seed in range(15):
            if time.time() >= deadline:
                break
            try:
                U_r, V_r, W_r, rank_r = _try_rank_reduction(26, U_cur, V_cur, W_cur, max_restarts=1, seed=seed, deadline=deadline)
                if U_r is not None:
                    best_rank = 26
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
            except Exception:
                continue
    
    # Final polish on the best result
    if time.time() < deadline:
        U_best = np.array(best_result["U"])
        V_best = np.array(best_result["V"])
        W_best = np.array(best_result["W"])
        U_pol, V_pol, W_pol, _ = _polish(U_best, V_best, W_best, max_iters=300, deadline=deadline)
        best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
    
    return best_result
# EVOLVE-BLOCK-END
