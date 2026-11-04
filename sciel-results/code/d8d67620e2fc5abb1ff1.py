import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # Safety margin for evaluator's 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

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
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

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

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Try to find lower rank solutions using structured search
    # The tensor has 56 non-zero entries. For rank R, we need R rank-1 terms.
    # Each rank-1 term (u, v, w) can cover multiple entries of T.
    # An entry (i,j,k) is covered if u[i,j]*v[j,k]*w[k,i] = 1.
    
    # Key insight: u[i,j] is non-zero for some (i,j), v[j,k] for some (j,k), w[k,i] for some (k,i).
    # A rank-1 term covers the "box" product of its support sets.
    # We need the union of all boxes to cover all 56 entries with product = 1.
    
    # For small R, we can enumerate possible support patterns.
    # Let's try R from 14 upward and see if we can find a solution.
    
    def find_structured_solution(target_rank):
        """Try to find a rank-target_rank solution using structured search."""
        if time.time() > deadline - 15:
            return None
            
        # For each rank-1 term, define its support:
        # S_U: set of (i,j) where u[i,j] != 0
        # S_V: set of (j,k) where v[j,k] != 0
        # S_W: set of (k,i) where w[k,i] != 0
        # The term covers entries (i,j,k) where (i,j) in S_U, (j,k) in S_V, (k,i) in S_W
        # and the product u[i,j]*v[j,k]*w[k,i] = 1.
        
        # Simplification: assume each support set is a single element (or small set).
        # For rank R, we need R terms. Let's try terms with single-element supports first.
        # A term with S_U={(i0,j0)}, S_V={(j0,k0)}, S_W={(k0,i0)} covers exactly entry (i0,j0,k0).
        # This gives the schoolbook solution with R=56.
        
        # To reduce rank, we need terms that cover multiple entries.
        # E.g., S_U={(0,0),(1,1)}, S_V={(0,0),(1,1)}, S_W={(0,0),(1,1)} covers 4 entries.
        
        # Let's use ALS with better initialization based on structured patterns.
        
        # Strategy: for each target_rank, try multiple structured initializations
        # and use ALS to refine.
        
        rng_struct = np.random.default_rng(target_rank * 1000 + 42)
        
        for init_strategy in range(6):
            if time.time() > deadline - 10:
                return None
                
            if init_strategy == 0:
                # Block diagonal initialization
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                block_size = max(1, 56 // target_rank)
                for r in range(target_rank):
                    # Assign blocks of entries to each term
                    start = r * block_size
                    end = min(start + block_size, 56)
                    for idx in range(start, end):
                        i = idx // (m*p)
                        jk = idx % (m*p)
                        j = jk // p
                        k = jk % p
                        U_init[i*m+j, r] = 1
                        V_init[j*p+k, r] = 1
                        W_init[k*n+i, r] = 1
            elif init_strategy == 1:
                # Random sparse initialization
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    # Randomly pick 2-4 entries for each factor
                    n_u = rng_struct.integers(2, 5)
                    n_v = rng_struct.integers(2, 5)
                    n_w = rng_struct.integers(2, 5)
                    u_idx = rng_struct.choice(n*m, n_u, replace=False)
                    v_idx = rng_struct.choice(m*p, n_v, replace=False)
                    w_idx = rng_struct.choice(p*n, n_w, replace=False)
                    U_init[u_idx, r] = rng_struct.choice([-1, 1], n_u)
                    V_init[v_idx, r] = rng_struct.choice([-1, 1], n_v)
                    W_init[w_idx, r] = rng_struct.choice([-1, 1], n_w)
            elif init_strategy == 2:
                # Based on Strassen-like patterns
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                # Use the rank-50 solution as a guide
                for r in range(target_rank):
                    # Mix columns from U50
                    cols = rng_struct.choice(min(50, target_rank+10), 2, replace=False)
                    U_init[:, r] = U50[:, cols[0]] + rng_struct.uniform(-0.1, 0.1, n*m)
                    V_init[:, r] = V50[:, cols[0]] + rng_struct.uniform(-0.1, 0.1, m*p)
                    W_init[:, r] = W50[:, cols[0]] + rng_struct.uniform(-0.1, 0.1, p*n)
            elif init_strategy == 3:
                # Kronecker-like structure
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    i0 = rng_struct.integers(0, n)
                    j0 = rng_struct.integers(0, m-1)
                    k0 = rng_struct.integers(0, p-1)
                    # 2x2x2 block
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                i = i0 + di
                                j = j0 + dj
                                k = k0 + dk
                                if i < n and j < m and k < p:
                                    U_init[i*m+j, r] += 1
                                    V_init[j*p+k, r] += 1
                                    W_init[k*n+i, r] += 1
            elif init_strategy == 4:
                # Orthogonal mixing of random vectors
                U_init = rng_struct.normal(0, 1, (n*m, target_rank))
                V_init = rng_struct.normal(0, 1, (m*p, target_rank))
                W_init = rng_struct.normal(0, 1, (p*n, target_rank))
                # Normalize
                for r in range(target_rank):
                    U_init[:, r] /= (np.linalg.norm(U_init[:, r]) + 1e-10)
                    V_init[:, r] /= (np.linalg.norm(V_init[:, r]) + 1e-10)
                    W_init[:, r] /= (np.linalg.norm(W_init[:, r]) + 1e-10)
            else:
                # Hybrid: some structured, some random
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                half = target_rank // 2
                # First half: structured
                for r in range(half):
                    i0 = r % n
                    j0 = (r // n) % m
                    k0 = (r // (n*m)) % p
                    U_init[i0*m+j0, r] = 1
                    V_init[j0*p+k0, r] = 1
                    W_init[k0*n+i0, r] = 1
                # Second half: random
                U_init[:, half:] = rng_struct.normal(0, 1, (n*m, target_rank-half))
                V_init[:, half:] = rng_struct.normal(0, 1, (m*p, target_rank-half))
                W_init[:, half:] = rng_struct.normal(0, 1, (p*n, target_rank-half))
            
            U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=100, tol=1e-10)
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-6:
                # Polish
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_sol, V_sol, W_sol, max_iters=20, tol=1e-12)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                if verify_decomp(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        
        return None

    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(T1, T2, T3, U, V, W, max_iters=100, tol=1e-10):
        R = U.shape[1]
        prev_residual = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
            if abs(prev_residual - residual) < 1e-12 * max(1.0, prev_residual):
                break
            prev_residual = residual
        return U, V, W
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(T1, T2, T3, U, V, W, max_iters=100, tol=1e-10):
        R = U.shape[1]
        prev_residual = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
            # Early stopping if no significant improvement
            if abs(prev_residual - residual) < 1e-12 * max(1.0, prev_residual):
                break
            prev_residual = residual
        return U, V, W

    def try_eliminate_column(U, V, W, col_idx):
        """Try to eliminate column col_idx by re-solving without it."""
        R = U.shape[1]
        if R <= 1:
            return None
        U_sub = np.delete(U, col_idx, axis=1)
        V_sub = np.delete(V, col_idx, axis=1)
        W_sub = np.delete(W, col_idx, axis=1)
        
        # Re-solve with R-1 columns
        U_new, V_new, W_new = als_solve(T1, T2, T3, U_sub, V_sub, W_sub, max_iters=30, tol=1e-8)
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        if np.allclose(recon, T, atol=1e-6):
            return U_new, V_new, W_new
        return None

    def greedy_column_elimination(U, V, W, max_rounds=8):
        """Iteratively try to eliminate columns that are redundant."""
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            R = U.shape[1]
            eliminated = False
            for col_idx in range(R - 1, -1, -1):
                result = try_eliminate_column(U, V, W, col_idx)
                if result is not None:
                    U, V, W = result
                    eliminated = True
                    break
            if not eliminated:
                break
        return U, V, W

    rng = np.random.default_rng(42)

    # Try structured search for lower ranks
    for target_rank in range(49, 35, -1):
        if time.time() > deadline - 15:
            break
        
        # Try structured approach first
        result = find_structured_solution(target_rank)
        if result is not None:
            U_sol, V_sol, W_sol = result
            U_sol, V_sol, W_sol = greedy_column_elimination(U_sol, V_sol, W_sol)
            final_rank = U_sol.shape[1]
            if verify_decomp(U_sol, V_sol, W_sol) and final_rank < best_rank:
                best_rank = final_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                if best_rank <= 40:
                    break
        
        # Also try random restarts
        for restart in range(4):
            if time.time() > deadline - 10:
                break
            if restart == 0:
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            elif restart == 1:
                Q = np.linalg.qr(rng.normal(0, 1, (target_rank, target_rank)))[0]
                U_init = U50[:, :target_rank] @ Q
                V_init = V50[:, :target_rank] @ Q
                W_init = W50[:, :target_rank] @ Q
            elif restart == 2:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            else:
                U_init = U50[:, :target_rank].copy() * rng.uniform(0.7, 1.3, (1, target_rank))
                V_init = V50[:, :target_rank].copy() * rng.uniform(0.7, 1.3, (1, target_rank))
                W_init = W50[:, :target_rank].copy() * rng.uniform(0.7, 1.3, (1, target_rank))

            U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_init, V_init, W_init, max_iters=80)
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-5:
                for _ in range(15):
                    if time.time() > deadline - 8:
                        break
                    U_sol, V_sol, W_sol = als_solve(T1, T2, T3, U_sol, V_sol, W_sol, max_iters=15, tol=1e-12)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

            if verify_decomp(U_sol, V_sol, W_sol):
                U_sol, V_sol, W_sol = greedy_column_elimination(U_sol, V_sol, W_sol)
                final_rank = U_sol.shape[1]
                if verify_decomp(U_sol, V_sol, W_sol) and final_rank < best_rank:
                    best_rank = final_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    if best_rank <= 40:
                        break

        if best_rank <= 40:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
