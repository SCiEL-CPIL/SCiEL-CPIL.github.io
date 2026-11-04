"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
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
        return U, V, W, rank

    def check_validity(U, V, W, tol=1e-7):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(r):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def build_linear_system_for_w(w, U, V):
        """Build A and b for solving W[w, :] given U, V."""
        r = U.shape[1]
        # A is (n*m*m*p, r)
        A = np.zeros((n * m * m * p, r))
        for c in range(r):
            col_u = U[:, c]  # (n*m,)
            col_v = V[:, c]  # (m*p,)
            A[:, c] = np.outer(col_u, col_v).flatten()
        
        # b is (n*m*m*p,)
        b = np.zeros(n * m * m * p)
        i_w = w % n
        k_w = w // n
        for j in range(m):
            u_idx = i_w * m + j
            v_idx = j * p + k_w
            b[u_idx * (m * p) + v_idx] = 1.0
        return A, b

    def solve_for_W(U, V):
        """Solve for W given U and V. Returns W or None if inconsistent."""
        r = U.shape[1]
        W = np.zeros((p * n, r))
        for w in range(p * n):
            A, b = build_linear_system_for_w(w, U, V)
            try:
                W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if len(residuals) > 0 and residuals[0] > 1e-5:
                return None
        return W

    def polish(U, V, W, max_iters=100, tol=1e-10):
        """Polish the decomposition by alternating updates."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update W given U, V
            W_new = np.zeros_like(W)
            for w in range(p * n):
                A, b = build_linear_system_for_w(w, U, V)
                try:
                    W_new[w, :], res, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return U, V, W
            W = W_new
            
            # Update U given V, W
            # For each (i, j), U[i*m+j, c] = sum over w,k of W[k*n+i, c] * V[j*p+k, c] * delta... 
            # Actually, the constraint is:
            # sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta(j1, j2)
            # Fixing i, j2, k, varying j1: this is linear in U[:, c] for c
            # Let's update U column by column or use lstsq
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j1 in range(m):
                    # Build system for U[i*m+j1, :]
                    # sum_c U[i*m+j1, c] * (V[j2*p+k, c] * W[k*n+i, c]) = delta(j1, j2) for all j2, k
                    A_u = np.zeros((m * p, r))
                    b_u = np.zeros(m * p)
                    for j2 in range(m):
                        for k in range(p):
                            row = j2 * p + k
                            A_u[row, :] = V[j2 * p + k, :] * W[k * n + i, :]
                            if j1 == j2:
                                b_u[row] = 1.0
                    try:
                        U_new[i * m + j1, :], res, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    except np.linalg.LinAlgError:
                        return U, V, W
            U = U_new
            
            # Update V given U, W
            V_new = np.zeros_like(V)
            for j2 in range(m):
                for k in range(p):
                    # sum_c (U[i*m+j1, c] * W[k*n+i, c]) * V[j2*p+k, c] = delta(j1, j2) for all i, j1
                    A_v = np.zeros((n * m, r))
                    b_v = np.zeros(n * m)
                    for i in range(n):
                        for j1 in range(m):
                            row = i * m + j1
                            A_v[row, :] = U[i * m + j1, :] * W[k * n + i, :]
                            if j1 == j2:
                                b_v[row] = 1.0
                    try:
                        V_new[j2 * p + k, :], res, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    except np.linalg.LinAlgError:
                        return U, V, W
            V = V_new
            
            # Check convergence
            if it > 0:
                diff = np.max(np.abs(U - U_new)) + np.max(np.abs(V - V_new)) + np.max(np.abs(W - W_new))
                if diff < tol:
                    break
        
        return U, V, W

    def try_rank_with_search(r, num_restarts=20, seed=42):
        """Try to find a valid decomposition with rank r."""
        rng = np.random.RandomState(seed)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            
            W = solve_for_W(U, V)
            if W is None:
                continue
            
            # Polish
            U, V, W = polish(U, V, W)
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None

    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try various ranks, starting from lower values
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But we know 4x4x4 has rank at most 49, and 4x4x9 is larger
    
    # Try structured approach: split B into [B1 | B2] where B1 is 4x4, B2 is 4x5
    # C = [A@B1 | A@B2]
    # We can potentially share some structure
    
    # Try ranks in a smart order
    ranks_to_try = []
    # Start from just above theoretical lower bound
    for r in range(36, 145):
        ranks_to_try.append(r)
    
    # Try some specific interesting ranks first
    priority_ranks = [49, 50, 54, 56, 60, 64, 72, 80, 81, 84, 90, 96, 100, 108, 112, 120, 128, 135, 140]
    ranks_to_try = sorted(set(priority_ranks + ranks_to_try))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if r >= best_rank:
            break
        
        # Use more restarts for lower ranks
        num_restarts = 30 if r < 80 else 15
        seed = 1000 + r
        
        result = try_rank_with_search(r, num_restarts=num_restarts, seed=seed)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break  # Found a valid one, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
