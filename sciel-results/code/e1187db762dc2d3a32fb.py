"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(result, atol=1e-8):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def build_system_for_W(U, V, R):
        """Build linear system A * W_flat = b for solving W given U, V.
        Variables: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
        Equations: for each (i, j, jp, k): sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j,jp)
        Vectorized construction.
        """
        num_eqs = n * m * m * p
        num_vars = P * R  # P = p*n
        
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        # Build b: 1 when j == jp
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * m * p + j * p + k
                    b[eq_idx] = 1.0
        
        # Vectorized A construction
        # For each (i, k, r): the block of coefficients for equations (i, j, jp, k) is U[i*m:(i+1)*m, r] outer V[k::p, r]
        for i in range(n):
            for k in range(p):
                u_block = U[i*m:(i+1)*m, :]  # shape (m, R)
                v_block = V[k::p, :]  # shape (m, R)
                # For each r, the (m x m) block is u_block[:, r] outer v_block[:, r]
                # We need to place these into A at the right positions
                for r in range(R):
                    var_idx = (k * n + i) * R + r
                    # Equations for this (i, k): eq indices are (i*m+j)*m*p + jp*p + k for j in [0,m), jp in [0,m)
                    # This is a block of m*m entries
                    u_col = u_block[:, r]  # length m
                    v_col = v_block[:, r]  # length m
                    block = np.outer(u_col, v_col)  # m x m
                    # Place into A: rows (i*m+j)*m*p + jp*p + k for j,jp in range(m)
                    row_indices = np.array([(i*m+j)*m*p + jp*p + k for j in range(m) for jp in range(m)])
                    A[row_indices, var_idx] = block.flatten()
        
        return A, b

    def solve_for_W(U, V, R, method='lstsq'):
        """Solve for W given U and V."""
        A, b = build_system_for_W(U, V, R)
        try:
            if method == 'lstsq':
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            else:
                # Try using pseudo-inverse for potentially better conditioning
                W_flat = np.linalg.pinv(A) @ b
            W = W_flat.reshape(P, R)
            return W
        except:
            return None

    def als_decompose(target, rank, max_iter=30, tol=1e-12):
        """ALS decomposition with structured W-solve and aggressive polish."""
        # Initialize with schoolbook-inspired structure: use first `rank` columns of schoolbook
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        # Add small random perturbation to break symmetry
        U += np.random.randn(N, rank) * 0.1
        V += np.random.randn(M, rank) * 0.1
        W += np.random.randn(P, rank) * 0.1
        
        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 20:
                break
            
            # Update W: fix U, V - use structured linear solve
            W_new = solve_for_W(U, V, rank)
            if W_new is not None:
                W = W_new
            
            # Update U: fix V, W
            H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V: fix U, W
            H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol and err < 1e-9:
                break
            prev_err = err
        
        # Aggressive polish: more iterations with tighter tolerance
        if time_remaining() > 30:
            for _ in range(30):
                if time_remaining() < 20:
                    break
                W_new = solve_for_W(U, V, rank)
                if W_new is not None:
                    W = W_new
                
                H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
                target_flat = target.reshape(N, M * P)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
                target_v = target.transpose(1, 0, 2).reshape(M, N * P)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Check if we're close enough
                T = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - target)
                if err < 1e-10:
                    break
        
        return U, V, W

    def try_rank(r, num_trials=5):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 30:
            return None
        for trial in range(num_trials):
            if time_remaining() < 25:
                break
            # Vary random seed per trial for diversity
            np.random.seed(42 + trial * 137)
            U, V, W = als_decompose(target, r, max_iter=40)
            result = {"U": U, "V": V, "W": W}
            if verify(result, atol=1e-7):
                return result
        return None

    # Try ranks below schoolbook, starting from 119 down
    # Focus on a narrower range first for speed
    for r in range(best_rank - 1, max(best_rank - 25, 95), -1):
        if time_remaining() < 40:
            break
        result = try_rank(r, num_trials=4)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # Continue searching for even lower ranks
            continue

    # If we have time, try a few more lower ranks
    if time_remaining() > 50:
        for r in range(max(best_rank - 1, 95), max(best_rank - 20, 90), -1):
            if time_remaining() < 35:
                break
            result = try_rank(r, num_trials=3)
            if result is not None:
                best = result
                best_rank = r
                print(f"Found decomposition with rank {r}")

    # Final verification
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    # Round to nearest half-integer if close (for cleaner output)
    U_out = np.array(best["U"])
    V_out = np.array(best["V"])
    W_out = np.array(best["W"])
    
    # Try rounding to nearest 0.5
    U_rounded = np.round(U_out * 2) / 2
    V_rounded = np.round(V_out * 2) / 2
    W_rounded = np.round(W_out * 2) / 2
    
    result_rounded = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    if verify(result_rounded, atol=1e-6):
        best = result_rounded

    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
# EVOLVE-BLOCK-END
