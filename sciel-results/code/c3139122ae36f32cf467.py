"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _als_step(T_A, T_B, T_C, U, V, W, nm, mp, pn, rank, tol=1e-10):
    """One ALS iteration updating U, V, W given the others."""
    # Update U: minimize ||T - sum_r V_r ⊗ W_r ⊗ U_r||
    # T_A shape (nm, mp*pn), U shape (nm, rank)
    # We need: U_new = T_A @ (V ⊙ W)^T @ ((V ⊙ W) @ (V ⊙ W)^T)^{-1}
    # where (V ⊙ W) has shape (mp*pn, rank) with element (j*p+k, r) = V[j*p+k,r]*W[k*n+i,r]... 
    # Actually let's use einsum properly
    
    # For U update: T_A[a, b*c] where a is nm-index, b is mp-index, c is pn-index
    # T = sum_r U[a,r] * V[b,r] * W[c,r]
    # So T_A[a, (b,c)] = sum_r U[a,r] * (V[b,r] * W[c,r])
    # Let M = V ⊗ W reshaped: M[b*c, r] = V[b,r] * W[c,r] but we need to be careful with indexing
    
    # Actually, let's compute the matricization properly
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # T_A[a, b*pn+c] = sum_r U[a,r] * (V[b,r]*W[c,r])
    # So define G = V * W element-wise then reshape: G[b,r]*W[c,r]... 
    
    # Let me use a cleaner approach:
    # T_A = T.reshape(nm, mp*pn)
    # For fixed r, the contribution to T_A is U[:,r] @ (V[:,r] * W[:,r]).T reshaped
    
    # Actually the standard ALS for CP decomposition:
    # T_A[a, bc] = sum_r U[a,r] * (V[b,r] W[c,r])
    # So T_A = U @ (V ⊙ W)^T where (V ⊙ W)[(b,c), r] = V[b,r] * W[c,r]
    
    # Build VW: shape (mp*pn, rank)
    # VW[b*pn + c, r] = V[b, r] * W[c, r]
    VW = np.einsum('br,cr->bcr', V, W)  # shape (mp, pn, rank)
    VW_flat = VW.transpose(1, 0, 1).reshape(pn, mp, rank)  # need (b,c,r) -> (c,b,r) for flattening? 
    # Wait, T_A[a, b*pn+c] so the second index is b*pn+c
    # VW_flat should have shape (mp*pn, rank) with index b*pn+c
    VW_flat = VW.transpose(0, 1, 2).reshape(mp * pn, rank)  # (b,c,r) -> (b*pn+c, r)
    
    # T_A = U @ VW_flat.T
    # U = T_A @ VW_flat @ (VW_flat.T @ VW_flat)^{-1}
    G = VW_flat.T @ VW_flat
    G_inv = np.linalg.pinv(G + tol * np.eye(rank))
    U_new = T_A @ VW_flat @ G_inv
    
    # For V update: T_B[b, a*pn+c] = sum_r V[b,r] * (U[a,r] * W[c,r])
    UW = np.einsum('ar,cr->acr', U_new, W)  # shape (nm, pn, rank)
    # T_B[b, a*pn+c] so second index is a*pn+c
    UW_flat = UW.transpose(1, 0, 2).reshape(pn, nm, rank)  # (c,a,r) -> need (a*pn+c, r)
    UW_flat = UW.transpose(0, 1, 2).reshape(nm * pn, rank)  # (a,c,r) -> (a*pn+c, r)
    
    G = UW_flat.T @ UW_flat
    G_inv = np.linalg.pinv(G + tol * np.eye(rank))
    V_new = T_B @ UW_flat @ G_inv
    
    # For W update: T_C[c, a*mp+b] = sum_r W[c,r] * (U[a,r] * V[b,r])
    UV = np.einsum('ar,br->abr', U_new, V_new)  # shape (nm, mp, rank)
    UV_flat = UV.transpose(0, 1, 2).reshape(nm * mp, rank)  # (a,b,r) -> (a*mp+b, r)
    
    G = UV_flat.T @ UV_flat
    G_inv = np.linalg.pinv(G + tol * np.eye(rank))
    W_new = T_C @ UV_flat @ G_inv
    
    return U_new, V_new, W_new

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    start_time = time.time()

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest quarter-integer
        U_r = np.round(U * 4) / 4
        V_r = np.round(V * 4) / 4
        W_r = np.round(W * 4) / 4
        if verify(U_r, V_r, W_r, atol=1e-9):
            return U_r, V_r, W_r
        
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    # Verify schoolbook is correct
    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()

    # Pre-compute matricizations for ALS
    # T_A: (nm, mp*pn) where T_A[a, b*pn+c] = T[a,b,c]
    T_A = T.reshape(n * m, m * p * p * n)
    # T_B: (mp, nm*pn) where T_B[b, a*pn+c] = T[a,b,c]
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    # T_C: (pn, nm*mp) where T_C[c, a*mp+b] = T[a,b,c]
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_decompose(U_init, V_init, W_init, max_iter=50, tol=1e-10):
        """Run ALS to refine the decomposition."""
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: T_A = U @ (V⊙W)^T
            # (V⊙W)[(b,c), r] = V[b,r] * W[c,r], shape (mp*pn, rank)
            MW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, rank)
            MW_flat = MW.reshape(m * p * p * n, rank)  # (b*pn+c, r)
            
            G = MW_flat.T @ MW_flat
            try:
                G_inv = np.linalg.pinv(G + tol * np.eye(rank))
                U_new = T_A @ MW_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update V: T_B = V @ (U⊙W)^T
            # (U⊙W)[(a,c), r] = U[a,r] * W[c,r], shape (nm*pn, rank)
            UW = np.einsum('ar,cr->acr', U_new, W)  # (nm, pn, rank)
            UW_flat = UW.reshape(n * m * p * n, rank)  # (a*pn+c, r)
            
            G = UW_flat.T @ UW_flat
            try:
                G_inv = np.linalg.pinv(G + tol * np.eye(rank))
                V_new = T_B @ UW_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Update W: T_C = W @ (U⊙V)^T
            # (U⊙V)[(a,b), r] = U[a,r] * V[b,r], shape (nm*mp, rank)
            UV = np.einsum('ar,br->abr', U_new, V_new)  # (nm, mp, rank)
            UV_flat = UV.reshape(n * m * m * p, rank)  # (a*mp+b, r)
            
            G = UV_flat.T @ UV_flat
            try:
                G_inv = np.linalg.pinv(G + tol * np.eye(rank))
                W_new = T_C @ UV_flat @ G_inv
            except np.linalg.LinAlgError:
                break

            # Check convergence
            delta = max(np.linalg.norm(U_new - U), np.linalg.norm(V_new - V), np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if delta < 1e-12:
                break

        return U, V, W

    # Strategy: Start from schoolbook and try to reduce rank
    # Try ranks from 71 down to 54
    target_ranks = list(range(rank_sch - 1, 53, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        
        # Multiple initialization strategies
        for restart in range(8):
            if time.time() > deadline:
                break
            
            if restart < 4:
                # Strategy 1: Random initialization
                np.random.seed(42 + restart)
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            else:
                # Strategy 2: Perturb schoolbook and project to lower rank
                # Take first target_rank columns of schoolbook + random perturbation
                np.random.seed(100 + restart)
                U_init = U_sch[:, :target_rank].copy() + 0.1 * np.random.randn(n * m, target_rank)
                V_init = V_sch[:, :target_rank].copy() + 0.1 * np.random.randn(m * p, target_rank)
                W_init = W_sch[:, :target_rank].copy() + 0.1 * np.random.randn(p * n, target_rank)
            
            # Run ALS
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iter=60)
            
            # Check if we found an exact solution
            if verify(U_sol, V_sol, W_sol, atol=1e-7):
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
            else:
                # Even if not exactly matching, try rounding the continuous solution
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break  # Found a valid lower-rank decomposition, stop searching

    # Final: ensure we return valid arrays
    U_out = best_U
    V_out = best_V
    W_out = best_W

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
