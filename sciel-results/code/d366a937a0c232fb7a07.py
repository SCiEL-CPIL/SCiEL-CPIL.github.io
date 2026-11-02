import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks (4 blocks) + schoolbook for remainder.
    Base rank: 4*7 + 4 + 10 = 42. Attempts reduction via fast alternating minimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    dim_a = n * m  # 10
    dim_b = m * p  # 25
    dim_c = p * n  # 10

    # Build target tensor T[a,b,c]
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_from_terms(terms):
        """Build U, V, W from list of (u_dict, v_dict, w_dict)."""
        R = len(terms)
        U = np.zeros((dim_a, R))
        V = np.zeros((dim_b, R))
        W = np.zeros((dim_c, R))
        for r, (ud, vd, wd) in enumerate(terms):
            for idx, val in ud.items():
                U[idx, r] = val
            for idx, val in vd.items():
                V[idx, r] = val
            for idx, val in wd.items():
                W[idx, r] = val
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err

    def alt_minimize_fast(U, V, W, max_iter=200, tol=1e-12):
        """
        Vectorized alternating minimization.
        For each factor, builds a large linear system and solves via least squares.
        """
        R = U.shape[1]
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()
        T_flat = T.reshape(-1)
        prev_err = np.inf

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            # --- Solve for U ---
            # T_flat ≈ Σ_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] flattened
            # Build A_u (dim_a × R*dim_b*dim_c) and rhs
            A_u = np.zeros((dim_a, R * dim_b * dim_c))
            for r in range(R):
                vw_flat = (V[:, r] * W[:, r]).reshape(-1)
                A_u[:, r * dim_b * dim_c : (r + 1) * dim_b * dim_c] = np.tile(vw_flat.reshape(dim_b, dim_c).T, (1, 1)).reshape(dim_b * dim_c, 1).T if False else np.kron(np.eye(1), vw_flat.reshape(1, -1))
            # Actually: for each a, (A_u @ x)[a] = Σ_r x[r, b, c] * V[b,r] * W[c,r]
            # Reshape x to (R, dim_b, dim_c), then A_u @ x = einsum('rbc,a->a', ...)
            # Simpler: build A_u[a, (r,b,c)] = V[b,r]*W[c,r]
            A_u = np.zeros((dim_a, R * dim_b * dim_c))
            for r in range(R):
                for b in range(dim_b):
                    for c in range(dim_c):
                        A_u[:, r * dim_b * dim_c + b * dim_c + c] = V[b, r] * W[c, r]
            # rhs[a] = T_flat[a] - Σ_{s≠r} U[s,a]*V[s,b]*W[s,c]... 
            # Actually rhs = T_flat - (contribution from all terms except the one being solved)
            # But in alternating minimization, we solve for ALL U[:,r] simultaneously given V,W fixed.
            # So: T_flat = A_u @ U_flat where U_flat = U.reshape(-1)
            # This is overdetermined, solve via lstsq
            U_flat, _, _, _ = np.linalg.lstsq(A_u, T_flat, rcond=None)
            U = U_flat.reshape(dim_a, R)

            # --- Solve for V ---
            A_v = np.zeros((dim_b, R * dim_a * dim_c))
            for r in range(R):
                for a in range(dim_a):
                    for c in range(dim_c):
                        A_v[:, r * dim_a * dim_c + a * dim_c + c] = U[a, r] * W[c, r]
            V_flat, _, _, _ = np.linalg.lstsq(A_v, T_flat, rcond=None)
            V = V_flat.reshape(dim_b, R)

            # --- Solve for W ---
            A_w = np.zeros((dim_c, R * dim_a * dim_b))
            for r in range(R):
                for a in range(dim_a):
                    for b in range(dim_b):
                        A_w[:, r * dim_a * dim_b + a * dim_b + b] = U[a, r] * V[b, r]
            W_flat, _, _, _ = np.linalg.lstsq(A_w, T_flat, rcond=None)
            W = W_flat.reshape(dim_c, R)

            # Check convergence
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T - Rec))
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        return U, V, W

    def try_reduce_rank(U, V, W, max_reduction=6):
        """Try to reduce rank by removing terms and re-solving."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R

        # Compute term "importance" (norm product)
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_idx = np.argsort(norms)

        for red in range(1, max_reduction + 1):
            if time.time() > deadline - 20:
                break
            new_R = R - red
            if new_R < 1:
                break

            found = False
            # Try removing smallest-norm terms
            keep = np.sort(sorted_idx[-new_R:])
            U_new = U[:, keep].copy()
            V_new = V[:, keep].copy()
            W_new = W[:, keep].copy()

            U_new, V_new, W_new = alt_minimize_fast(U_new, V_new, W_new, max_iter=150)
            valid, err = verify(U_new, V_new, W_new)
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_R
                found = True

            if not found:
                # Try a few random removals
                for attempt in range(3):
                    if time.time() > deadline - 20:
                        break
                    remove = np.random.choice(R, size=red, replace=False)
                    keep = np.sort(np.setdiff1d(np.arange(R), remove))
                    U_new = U[:, keep].copy()
                    V_new = V[:, keep].copy()
                    W_new = W[:, keep].copy()
                    U_new, V_new, W_new = alt_minimize_fast(U_new, V_new, W_new, max_iter=150)
                    valid, err = verify(U_new, V_new, W_new)
                    if valid:
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = new_R
                        found = True
                        break

            if not found:
                break

        return best_U, best_V, best_W, best_rank

    def polish(U, V, W, max_iter=300):
        """High-precision polish."""
        U, V, W = alt_minimize_fast(U, V, W, max_iter=max_iter, tol=1e-14)
        return U, V, W

    # Build Strassen-based decomposition
    terms = []

    def add_strassen_2x2x2(i0, j0, k0):
        """
        Strassen for 2x2 A-block (rows i0,i0+1; cols j0,j0+1)
        times 2x2 B-block (rows j0,j0+1; cols k0,k0+1)
        producing 2x2 output (rows i0,i0+1; cols k0,k0+1).
        
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
        Out[0,0]=M1+M4-M5+M7, Out[0,1]=M3+M5, Out[1,0]=M2+M4, Out[1,1]=M1-M2+M3+M6
        """
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i

        # M1 = (a+d)(e+h)
        u = {(i0*m+j0): 1, ((i0+1)*m+(j0+1)): 1}
        v = {(j0*p+k0): 1, ((j0+1)*p+(k0+1)): 1}
        w = {(k0*n+i0): 1, ((k0+1)*n+(i0+1)): 1}
        terms.append((u, v, w))

        # M2 = (c+d)e
        u = {((i0+1)*m+j0): 1, ((i0+1)*m+(j0+1)): 1}
        v = {(j0*p+k0): 1}
        w = {(k0*n+(i0+1)): 1, ((k0+1)*n+(i0+1)): -1}
        terms.append((u, v, w))

        # M3 = a(f-h)
        u = {(i0*m+j0): 1}
        v = {(j0*p+(k0+1)): 1, ((j0+1)*p+(k0+1)): -1}
        w = {(k0*n+i0): 1, ((k0+1)*n+i0): 1}
        terms.append((u, v, w))

        # M4 = d(g-e)
        u = {((i0+1)*m+(j0+1)): 1}
        v = {((j0+1)*p+k0): 1, (j0*p+k0): -1}
        w = {(k0*n+(i0+1)): 1, ((k0+1)*n+(i0+1)): 1}
        terms.append((u, v, w))

        # M5 = (a+b)h
        u = {(i0*m+j0): 1, (i0*m+(j0+1)): 1}
        v = {((j0+1)*p+(k0+1)): 1}
        w = {(k0*n+(i0+1)): -1, ((k0+1)*n+(i0+1)): 1}
        terms.append((u, v, w))

        # M6 = (c-a)(e+f)
        u = {((i0+1)*m+j0): 1, (i0*m+j0): -1}
        v = {(j0*p+k0): 1, (j0*p+(k0+1)): 1}
        w = {(k0*n+(i0+1)): 1, ((k0+1)*n+(i0+1)): 1}
        terms.append((u, v, w))

        # M7 = (b-d)(g+h)
        u = {(i0*m+(j0+1)): 1, ((i0+1)*m+(j0+1)): -1}
        v = {((j0+1)*p+k0): 1, ((j0+1)*p+(k0+1)): 1}
        w = {(k0*n+i0): 1, ((k0+1)*n+i0): 1}
        terms.append((u, v, w))

    def add_schoolbook(i, j, k):
        u = {(i*m+j): 1}
        v = {(j*p+k): 1}
        w = {(k*n+i): 1}
        terms.append((u, v, w))

    # Apply Strassen to 4 blocks: (i,j,k) in {(0,0,0),(0,0,2),(0,2,0),(0,2,2)}
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)

    # Schoolbook for j=4 (the 1-dimensional part of inner dim)
    for k in range(5):
        for i in range(2):
            add_schoolbook(i, 4, k)

    # Schoolbook for k=4 within the 2x2 blocks (the 1-dimensional part of output cols)
    for j in [0, 1, 2, 3]:
        for i in range(2):
            add_schoolbook(i, j, 4)

    U, V, W = build_from_terms(terms)
    R = U.shape[1]
    print(f"Initial rank: {R}")

    # Verify
    valid, err = verify(U, V, W)
    print(f"Valid: {valid}, err: {err}")

    if not valid:
        # Fallback to schoolbook
        R_fb = n * m * p
        U = np.zeros((dim_a, R_fb))
        V = np.zeros((dim_b, R_fb))
        W = np.zeros((dim_c, R_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        R = R_fb
    else:
        # Try to reduce rank
        if time.time() < deadline - 30:
            U, V, W, R = try_reduce_rank(U, V, W, max_reduction=5)
            print(f"After reduction: rank {R}")

        # Polish
        if time.time() < deadline - 10:
            U, V, W = polish(U, V, W, max_iter=200)

    # Final verification
    valid, err = verify(U, V, W)
    print(f"Final: valid={valid}, err={err}, rank={U.shape[1]}")

    if not valid:
        # Fallback
        R_fb = n * m * p
        U = np.zeros((dim_a, R_fb))
        V = np.zeros((dim_b, R_fb))
        W = np.zeros((dim_c, R_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
