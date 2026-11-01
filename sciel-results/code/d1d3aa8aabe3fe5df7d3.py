"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def time_left():
        return deadline - time.time()

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-6, err

    def strassen_42():
        """Fallback: 6 independent Strassen blocks, rank 42."""
        rank = 42
        U = np.zeros((4, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        # Standard Strassen 7 terms for 2x2x2
        # A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # B indices (local 2x2): 0=b00, 1=b01, 2=b10, 3=b11
        # C indices (local, W[k*n+i]): 0=c00, 1=c01, 2=c10, 3=c11
        strassen_terms = [
            # (U entries, V entries, W entries)
            # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2 = (a10+a11)b00 -> c10 += 1
            ([(2,1),(3,1)], [(0,1)], [(2,1)]),
            # M3 = a00(b01-b11) -> c01 += 1
            ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
            # M4 = a11(b10-b00) -> c00 += 1, c10 += 1
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6 = (a10-a00)(b00+b01) -> c11 += 1
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7 = (a01-a11)(b10+b11) -> c00 -= 1, c11 += 1
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(3,1)]),
        ]
        for block in range(6):
            for s in range(7):
                col = block * 7 + s
                u_e, v_e, w_e = strassen_terms[s]
                for idx, coeff in u_e:
                    U[idx, col] = coeff
                for idx, coeff in v_e:
                    V[idx + block * 4, col] = coeff
                for idx, coeff in w_e:
                    W[idx + block * 4, col] = coeff
        return U, V, W

    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=2000):
        """
        Try to reduce the rank by zeroing out columns and re-optimizing.
        Uses alternating minimization on the remaining columns.
        """
        r = target_rank
        U = U_init[:, :r].copy()
        V = V_init[:, :r].copy()
        W = W_init[:, :r].copy()

        for it in range(max_iters):
            if time_left() < 5:
                break
            # Update U: for each row a, solve least squares
            # T[a,:,:] = U[a,:] @ (V[:,:,None] * W[:,:,None]).transpose(1,2,0)
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed U: T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
            # Reshape: T[a] = U[a] @ M where M[r, (b,c)] = V[b,r]*W[c,r]
            M_VW = np.einsum('br,cr->bc r', V, W)  # shape (24, 24, r)
            M_VW_flat = M_VW.reshape(576, r)
            # T_flat[a] = U[a] @ M_VW_flat^T
            T_flat = T.reshape(4, 576)
            # Solve: T_flat = U @ M_VW_flat.T  =>  U = T_flat @ pinv(M_VW_flat.T)
            # M_VW_flat.T has shape (r, 576)
            try:
                U_new = T_flat @ np.linalg.pinv(M_VW_flat.T)
            except:
                break
            U = U_new

            # Update V: T[:,b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
            M_UW = np.einsum('ar,cr->ac r', U, W)  # shape (4, 24, r)
            M_UW_flat = M_UW.reshape(96, r)
            T_flat_v = T.reshape(24, 96).T  # shape (96, 24) -> we want T[:,b,:] for each b
            # T[:,b,:] has shape (4,24), flatten to (96,)
            # T_flat_v[96_idx, b] where 96_idx = a*24+c
            # T[:,b,:] = V[b] @ M_UW_flat.T  => V = T_flat_v.T @ pinv(M_UW_flat.T)
            # T_flat_v.T has shape (24, 96)
            try:
                V_new = T_flat_v.T @ np.linalg.pinv(M_UW_flat.T)
            except:
                break
            V = V_new

            # Update W: T[:,:,c] = sum_r W[c,r] (U[:,r] outer V[:,r])
            M_UV = np.einsum('ar,br->ab r', U, V)  # shape (4, 24, r)
            M_UV_flat = M_UV.reshape(96, r)
            T_flat_w = T.reshape(96, 24)  # T[:,:,c] flattened: (a*24+b, c)
            # T_flat_w[96_idx, c] = W[c] @ M_UV_flat.T
            # W = T_flat_w.T @ pinv(M_UV_flat.T)
            try:
                W_new = T_flat_w.T @ np.linalg.pinv(M_UV_flat.T)
            except:
                break
            W = W_new

            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-9:
                break

        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return U, V, W, err

    def random_init(rank):
        """Random initialization for rank-r decomposition."""
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(24, rank) * 0.5
        W = np.random.randn(24, rank) * 0.5
        return U, V, W

    def search_rank(target_rank, n_restarts=10, max_iters=3000):
        """Search for a valid rank-target_rank decomposition."""
        best_err = float('inf')
        best_result = None

        for restart in range(n_restarts):
            if time_left() < 10:
                break

            # Initialize near the rank-42 solution for better convergence
            if target_rank <= 42:
                U42, V42, W42 = strassen_42()
                # Take first target_rank columns and add small random perturbation
                U = U42[:, :target_rank].copy() + np.random.randn(4, target_rank) * 0.01
                V = V42[:, :target_rank].copy() + np.random.randn(24, target_rank) * 0.01
                W = W42[:, :target_rank].copy() + np.random.randn(24, target_rank) * 0.01
            else:
                U, V, W = random_init(target_rank)

            # Run alternating minimization
            U, V, W, err = try_rank_reduction(U, V, W, target_rank, max_iters=max_iters)

            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())

            if best_err < 1e-9:
                break

            # Also try a few more random inits
            if restart % 3 == 2:
                U, V, W = random_init(target_rank)
                U, V, W, err2 = try_rank_reduction(U, V, W, target_rank, max_iters=max_iters // 2)
                if err2 < best_err:
                    best_err = err2
                    best_result = (U.copy(), V.copy(), W.copy())
                if best_err < 1e-9:
                    break

        return best_result, best_err

    # Strategy: Try to find rank < 42 using global search
    # Theoretical lower bound: max(4, 24, 24) = 24, but realistically much higher
    # Try ranks from 30 to 41 (below 42)

    best_overall = None
    best_rank = 42
    best_err = float('inf')

    # First get the rank-42 solution as baseline
    U42, V42, W42 = strassen_42()
    valid42, err42 = verify(U42, V42, W42)
    if valid42:
        best_overall = (U42, V42, W42)
        best_rank = 42
        best_err = err42

    # Try to find lower rank decompositions
    # Start from rank 41 down to 30
    for target_rank in range(41, 29, -1):
        if time_left() < 20:
            break

        result, err = search_rank(target_rank, n_restarts=5, max_iters=2000)
        if result is not None and err < 1e-6:
            U, V, W = result
            valid, final_err = verify(U, V, W)
            if valid:
                best_overall = (U, V, W)
                best_rank = target_rank
                best_err = final_err
                print(f"Found rank {target_rank} decomposition with error {final_err:.2e}")
                break
        elif result is not None and err < best_err:
            # Track best even if not converged, in case we can polish
            pass

    # If we found something better, return it; otherwise return rank 42
    if best_overall is not None:
        U, V, W = best_overall
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Ultimate fallback: schoolbook rank 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
# EVOLVE-BLOCK-END
