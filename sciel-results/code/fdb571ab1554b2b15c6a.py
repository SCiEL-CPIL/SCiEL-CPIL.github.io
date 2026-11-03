import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = 2, 5, 5
    U_dim, V_dim, W_dim = n * m, m * p, p * n  # 10, 25, 10

    # Build the target matmul tensor T[u,v,w]=1 iff u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def time_left():
        return deadline - time.time()

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return bool(np.allclose(prod, T, atol=atol))

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    # ---------- Rank-46 fallback: Strassen on 2x2x2 blocks + schoolbook edges ----------
    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00+A11)(B00+B11)
            U[0 * m + j_off][c] += 1; U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1; V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1; W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2 = (A10+A11)B00
            U[1 * m + j_off][c] += 1; U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1; W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3 = A00(B01-B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1; V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1; W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4 = A11(B10-B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1; V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1; W[k_off * n + 1][c] += 1
            c += 1
            # P5 = (A00+A01)B11
            U[0 * m + j_off][c] += 1; U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1; W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6 = (A10-A00)(B00+B01)
            U[1 * m + j_off][c] += 1; U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1; V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7 = (A01-A11)(B10+B11)
            U[0 * m + (j_off + 1)][c] += 1; U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1; V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1; V[4 * p + k][col] = 1; W[k * n + i][col] = 1; col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1; V[j * p + 4][col] = 1; W[4 * n + i][col] = 1; col += 1
        for i in range(2):
            U[i * m + 4][col] = 1; V[4 * p + 4][col] = 1; W[4 * n + i][col] = 1; col += 1

        return U, V, W

    # ---------- Middle-product block decomposition rank-10 init ----------
    def build_rank10_init():
        rank = 10
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        for r in range(5):
            for i in range(2):
                for k in range(5):
                    U[i * m + r, r] = 1.0
                    V[r * p + k, r] = 1.0
                    W[k * n + i, r + (5 if i == 1 else 0)] = 1.0
        return U, V, W

    # ---------- Gradient descent with momentum + periodic half-integer snapping ----------
    def optimize(U_init, V_init, W_init, target_rank, max_iter=2000, lr=0.02,
                 momentum=0.9, snap_every=60):
        R = target_rank
        U = U_init[:, :R].astype(float).copy()
        V = V_init[:, :R].astype(float).copy()
        W = W_init[:, :R].astype(float).copy()
        vU = np.zeros_like(U); vV = np.zeros_like(V); vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        for it in range(max_iter):
            if time_left() < 8:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            U -= vU; V -= vV; W -= vW
            loss = float(np.sum(diff * diff))
            if loss < prev_loss - 1e-12:
                prev_loss = loss; no_improve = 0
            else:
                no_improve += 1
                if no_improve > 60:
                    break
            if loss < 1e-10:
                break
            if it % snap_every == snap_every - 1:
                ok, Us, Vs, Ws = try_snap_validate(U, V, W)
                if ok:
                    return Us, Vs, Ws
        ok, Us, Vs, Ws = try_snap_validate(U, V, W)
        if ok:
            return Us, Vs, Ws
        return None, None, None

    # ---------- Alternate least squares ----------
    def solve_for_U(V, W, R):
        U = np.zeros((U_dim, R))
        for i in range(n):
            for j in range(m):
                A_block = np.zeros((p, R)); b_block = np.ones(p)
                for k in range(p):
                    A_block[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                except np.linalg.LinAlgError:
                    U[i * m + j, :] = 0
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((V_dim, R))
        for j in range(m):
            for k in range(p):
                A_block = np.zeros((n, R)); b_block = np.ones(n)
                for i in range(n):
                    A_block[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                except np.linalg.LinAlgError:
                    V[j * p + k, :] = 0
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((W_dim, R))
        for k in range(p):
            for i in range(n):
                A_block = np.zeros((m, R)); b_block = np.ones(m)
                for j in range(m):
                    A_block[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                except np.linalg.LinAlgError:
                    W[k * n + i, :] = 0
        return W

    def alternating(U, V, W, cycles=60):
        R = U.shape[1]
        for _ in range(cycles):
            if time_left() < 8:
                break
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
        return U, V, W

    # ---------- NEW: Local rank reduction via null-space perturbation ----------
    def try_local_reduction(U, V, W, max_retries=50):
        """
        Structural approach: start from a valid rank-R decomposition and try to
        find a linear dependency among the rank-1 terms. If we can express one
        term as a combination of others, we can drop it and reduce rank by 1.
        
        We do this by checking if the columns of the concatenated tensor slices
        have a null vector that corresponds to a valid rank reduction.
        """
        R = U.shape[1]
        if R <= 1:
            return None, None, None
        
        # The tensor T = sum_r U[:,r] V[:,r] W[:,r]^T (outer product form)
        # A rank reduction by 1 exists if there are coefficients c_r (not all zero)
        # such that sum_r c_r U[:,r] V[:,r] W[:,r]^T = 0
        # This is equivalent to: for all (a,b,c), sum_r c_r U[a,r] V[b,r] W[c,r] = 0
        
        # We can frame this as: find c in null space of a certain linear map.
        # Map: c -> tensor sum_r c_r U[:,r] V[:,r] W[:,r]^T
        # The tensor has U_dim*V_dim*W_dim entries.
        # We want c such that this tensor is 0.
        
        # Build the linear system: for each (a,b,c), sum_r c_r * U[a,r]*V[b,r]*W[c,r] = 0
        # This is a system of U_dim*V_dim*W_dim equations in R unknowns.
        # But many equations are redundant. Let's use a subset.
        
        # Actually, a cleaner approach: the rank of the tensor decomposition is R.
        # We can check if we can reduce by looking at the matricizations.
        
        # Alternative: use the fact that if we have a valid decomposition, we can
        # try to find a linear relation by solving a homogeneous system.
        
        # Let's build a matrix M where each row corresponds to a (a,b) pair, and
        # M[(a,b), r] = U[a,r] * V[b,r]. Then for a fixed c, the condition is:
        # sum_r c_r * U[a,r] V[b,r] W[c,r] = 0 for all (a,b).
        # This means: for each c, the vector w_c = W[:,c] (length R) must satisfy
        # M @ (c * w_c) = 0, i.e., M @ (c * w_c) = 0.
        # But c is what we're solving for, and w_c depends on c through W.
        
        # Let me try a different approach: iterative rank reduction.
        # Pick a random column r0 to eliminate. Try to express term r0 as a combo of others.
        
        for attempt in range(max_retries):
            if time_left() < 5:
                break
            
            # Pick a random column to try to eliminate
            r0 = np.random.randint(R)
            
            # We want to find coefficients d_r for r != r0 such that:
            # U[:,r0] V[:,r0] W[:,r0]^T = sum_{r != r0} d_r U[:,r] V[:,r] W[:,r]^T
            
            # This is a system of U_dim * V_dim * W_dim equations in R-1 unknowns.
            # Let's use a subset of equations for speed.
            
            # Build the linear system using random samples of (a,b,c)
            n_samples = min(200, U_dim * V_dim * W_dim)
            A_sys = np.zeros((n_samples, R - 1))
            b_sys = np.zeros(n_samples)
            
            indices = np.random.randint(0, U_dim * V_dim * W_dim, size=n_samples)
            a_idx = indices // (V_dim * W_dim)
            bc_idx = indices % (V_dim * W_dim)
            b_idx = bc_idx // W_dim
            c_idx = bc_idx % W_dim
            
            for s in range(n_samples):
                a_s = a_idx[s]
                b_s = b_idx[s]
                c_s = c_idx[s]
                
                b_sys[s] = U[a_s, r0] * V[b_s, r0] * W[c_s, r0]
                
                col = 0
                for r in range(R):
                    if r == r0:
                        continue
                    A_sys[s, col] = U[a_s, r] * V[b_s, r] * W[c_s, r]
                    col += 1
            
            try:
                d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is good
            residual = np.sum((A_sys @ d - b_sys) ** 2)
            if residual > 1e-6:
                continue
            
            # Build new matrices with the column eliminated
            keep = [r for r in range(R) if r != r0]
            U_new = np.zeros((U_dim, R - 1))
            V_new = np.zeros((V_dim, R - 1))
            W_new = np.zeros((W_dim, R - 1))
            
            for new_r, old_r in enumerate(keep):
                U_new[:, new_r] = U[:, old_r]
                V_new[:, new_r] = V[:, old_r]
                W_new[:, new_r] = W[:, old_r]
            
            # Add the contribution of the eliminated term to the remaining terms
            for new_r, old_r in enumerate(keep):
                U_new[:, new_r] += d[new_r] * U[:, r0]
                V_new[:, new_r] += d[new_r] * V[:, r0]
                W_new[:, new_r] += d[new_r] * W[:, r0]
            
            # Validate
            if validate(U_new, V_new, W_new, atol=1e-5):
                ok, Us, Vs, Ws = try_snap_validate(U_new, V_new, W_new)
                if ok:
                    return Us, Vs, Ws
                else:
                    return U_new, V_new, W_new
        
        return None, None, None

    # ---------- Search loop ----------
    U_best, V_best, W_best = build_rank46()
    if not validate(U_best, V_best, W_best):
        # schoolbook fallback (rank 50)
        rank = n * m * p
        U_best = np.zeros((U_dim, rank)); V_best = np.zeros((V_dim, rank)); W_best = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_best[i * m + j, col] = 1.0
                    V_best[j * p + k, col] = 1.0
                    W_best[k * n + i, col] = 1.0
                    col += 1
    best_rank = U_best.shape[1]

    # Try local rank reduction first (from rank 46)
    if time_left() > 30:
        np.random.seed(42)
        res = try_local_reduction(U_best, V_best, W_best, max_retries=100)
        if res[0] is not None:
            U_best, V_best, W_best = res
            best_rank = U_best.shape[1]

    # Try rank 10 and 45 using both optimizers
    for target_rank in (10, 45):
        if best_rank <= target_rank:
            continue
        if time_left() < 25:
            break
        for restart in range(6):
            if time_left() < 12:
                break
            np.random.seed(1000 + target_rank * 10 + restart)
            if target_rank == 10:
                U0, V0, W0 = build_rank10_init()
            else:
                U0, V0, W0 = build_rank46()[:, :target_rank]
            U0 = U0 + np.random.randn(*U0.shape) * 0.05
            V0 = V0 + np.random.randn(*V0.shape) * 0.05
            W0 = W0 + np.random.randn(*W0.shape) * 0.05

            # Gradient family
            res = optimize(U0, V0, W0, target_rank, max_iter=1500, lr=0.02,
                           momentum=0.9, snap_every=60)
            if res[0] is not None:
                U_best, V_best, W_best = res
                best_rank = target_rank
                break

            if time_left() < 12:
                break
            # ALS family
            Ua, Va, Wa = alternating(U0, V0, W0, cycles=60)
            ok, Us, Vs, Ws = try_snap_validate(Ua, Va, Wa)
            if ok:
                U_best, V_best, W_best = Us, Vs, Ws
                best_rank = target_rank
                break

    # Try local reduction again if we still have rank 46
    if best_rank >= 45 and time_left() > 20:
        np.random.seed(123)
        res = try_local_reduction(U_best, V_best, W_best, max_retries=100)
        if res[0] is not None:
            U_best, V_best, W_best = res
            best_rank = U_best.shape[1]

    def to_list(mat):
        return mat.tolist()

    return {
        "U": to_list(U_best),
        "V": to_list(V_best),
        "W": to_list(W_best)
    }
