import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the rank-32 Strassen-based decomposition
    rank32 = 32
    U = [[0] * rank32 for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank32 for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank32 for _ in range(p * n)]   # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M2
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        # M6
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    assert col == rank32, f"col={col}"

    # Build tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    N = len(tensor_entries)  # 36

    import numpy as np

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    ta = np.array([e[0] for e in tensor_entries])
    tb = np.array([e[1] for e in tensor_entries])
    tc = np.array([e[2] for e in tensor_entries])

    # Precompute per-entry index arrays for fast residual
    # For each entry t, we need U[ta[t],:], V[tb[t],:], W[tc[t],:]
    U_t = U_np[ta]  # (N, rank32)
    V_t = V_np[tb]  # (N, rank32)
    W_t = W_np[tc]  # (N, rank32)

    def residual_from(Ur, Vr, Wr, drop_cols):
        """Compute residual given factors with columns to drop removed."""
        # Ur, Vr, Wr have shape (N, R) where R = rank32 - len(drop_cols)
        vals = np.einsum('ij,ij,ij->i', Ur, Vr, Wr)
        return float(np.sum((vals - 1.0) ** 2))

    # Verify rank-32 is exact
    full_res = residual_from(U_t, V_t, W_t, None)
    assert abs(full_res) < 1e-20, f"rank32 residual {full_res}"

    def drop_columns(Ur, Vr, Wr, drop_cols):
        keep = [c for c in range(rank32) if c not in drop_cols]
        return Ur[:, keep], Vr[:, keep], Wr[:, keep]

    # Strategy: try to find rank-31 by dropping one column and adjusting
    # The residual after dropping column d is the "error vector" e_t = -U[ta[t],d]*V[tb[t],d]*W[tc[t],d]
    # We need to find small perturbations to the remaining 31 columns to cancel this.
    #
    # Key: for a rank-31 solution to exist, there must be a column d such that the
    # error vector e lies in the span achievable by perturbing the other 31 columns.
    #
    # Approach: for each drop d, do coordinate descent to minimize residual.

    def try_rank31(drop_col):
        """Try to find rank-31 solution by dropping column drop_col and optimizing."""
        keep = [c for c in range(rank32) if c != drop_col]
        Uc = U_np[:, keep].copy()  # (4, 31)
        Vc = V_np[:, keep].copy()  # (18, 31)
        Wc = W_np[:, keep].copy()  # (18, 31)

        # Precompute per-entry
        Uc_t = Uc[ta]  # (N, 31)
        Vc_t = Vc[tb]
        Wc_t = Wc[tc]

        # Current residual (should be nonzero)
        vals = np.einsum('ij,ij,ij->i', Uc_t, Vc_t, Wc_t)
        errs = vals - 1.0  # (N,)
        res = float(np.sum(errs ** 2))
        if res < 1e-24:
            return True, Uc, Vc, Wc

        # Coordinate descent: for each factor entry, compute optimal value
        # The problem is sum_t (sum_r Uc[ta[t],r]*Vc[tb[t],r]*Wc[tc[t],r] - 1)^2
        # For a fixed factor, say Uc[a, r], the contribution to the sum is:
        #   sum_{t: ta[t]==a} (pred_t - 1) * Vc[tb[t],r] * Wc[tc[t],r] * Uc[a,r]
        # plus terms independent of Uc[a,r]
        # So optimal Uc[a,r] = - sum_{t:ta[t]==a}(pred_t-1)*Vc[tb[t],r]*Wc[tc[t],r] / sum_{t:ta[t]==a}(Vc[tb[t],r]*Wc[tc[t],r])^2
        #
        # This is expensive to recompute each time. Use gradient descent instead.

        # Gradient-based: use Adam-like update with full gradient
        # grad_Uc[a,r] = 2 * sum_{t:ta[t]==a} errs[t] * Vc[tb[t],r] * Wc[tc[t],r]

        # Precompute which entries have each a value
        a_groups = {}
        for t in range(N):
            a = ta[t]
            if a not in a_groups:
                a_groups[a] = []
            a_groups[a].append(t)

        # For speed, precompute for each a the list of (t, r) contributions
        # Actually let's just do vectorized gradient

        def compute_grads(Uc, Vc, Wc, errs):
            # grad_Uc[a,r] = 2 * sum_{t:ta[t]==a} errs[t] * Vc[tb[t],r] * Wc[tc[t],r]
            # = 2 * sum_t errs[t] * delta(ta[t],a) * Vc[tb[t],r] * Wc[tc[t],r]
            # Build matrix G_U[a, r] = sum_t errs[t]*delta(ta[t],a)*Vc[tb[t],r]*Wc[tc[t],r]
            # For each r: G_U[:, r] = sum_t errs[t]*delta(ta[t],:)* (Vc[tb[t],r]*Wc[tc[t],r])
            # = for each r, vector over a: sum_t errs[t] * [ta[t]==a] * Vc[tb[t],r] * Wc[tc[t],r]

            # Vectorized:
            # prodVW = Vc[tb] * Wc[tc]  # (N, 31)
            # For each r, G_U[:, r] = sum_t errs[t] * onehot(ta[t]) * prodVW[t, r]
            # = (errs * prodVW) aggregated by ta

            prodVW = Vc_t * Wc_t  # (N, 31)
            prodUV = Uc_t * Vc_t
            prodUW = Uc_t * Wc_t

            # G_U[a, r] = sum_t errs[t] * (ta[t]==a) * prodVW[t, r]
            # Use scatter-add
            G_U = np.zeros((n*m, 31))
            np.add.at(G_U, (ta, np.tile(np.arange(31), (N,1)).T), (errs[:,None] * prodVW).T)

            G_V = np.zeros((m*p, 31))
            np.add.at(G_V, (tb, np.tile(np.arange(31), (N,1)).T), (errs[:,None] * prodUV).T)

            G_W = np.zeros((p*n, 31))
            np.add.at(G_W, (tc, np.tile(np.arange(31), (N,1)).T), (errs[:,None] * prodUW).T)

            return 2*G_U, 2*G_V, 2*G_W

        # Adam optimizer
        lr = 0.02
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mU = np.zeros_like(Uc); vU = np.zeros_like(Uc)
        mV = np.zeros_like(Vc); vV = np.zeros_like(Vc)
        mW = np.zeros_like(Wc); vW = np.zeros_like(Wc)
        t = 0
        best_res = res
        best_Uc, best_Vc, best_Wc = Uc.copy(), Vc.copy(), Wc.copy()

        # We'll do up to ~2000 iterations, checking time
        max_iter = 3000
        for it in range(max_iter):
            if time.time() > deadline:
                break
            t += 1
            vals = np.einsum('ij,ij,ij->i', Uc_t, Vc_t, Wc_t)
            errs = vals - 1.0
            res = float(np.sum(errs ** 2))
            if res < best_res:
                best_res = res
                best_Uc, best_Vc, best_Wc = Uc.copy(), Vc.copy(), Wc.copy()
                if best_res < 1e-14:
                    return True, best_Uc, best_Vc, best_Wc

            gU, gV, gW = compute_grads(Uc, Vc, Wc, errs)

            # Adam update
            mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
            mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
            mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
            mUh = mU/(1-beta1**t); vUh = vU/(1-beta2**t)
            mVh = mV/(1-beta1**t); vVh = vV/(1-beta2**t)
            mWh = mW/(1-beta1**t); vWh = vW/(1-beta2**t)

            Uc -= lr*mUh/(np.sqrt(vUh)+eps)
            Vc -= lr*mVh/(np.sqrt(vVh)+eps)
            Wc -= lr*mWh/(np.sqrt(vWh)+eps)

            # Update precomputed
            Uc_t = Uc[ta]; Vc_t = Vc[tb]; Wc_t = Wc[tc]

        return best_res < 1e-12, best_Uc, best_Vc, best_Wc

    # Try each drop column
    found = False
    for drop in range(rank32):
        if time.time() > deadline - 5:
            break
        success, Uc, Vc, Wc = try_rank31(drop)
        if success:
            # Verify
            Uc_t = Uc[ta]; Vc_t = Vc[tb]; Wc_t = Wc[tc]
            vals = np.einsum('ij,ij,ij->i', Uc_t, Vc_t, Wc_t)
            final_res = float(np.sum((vals-1.0)**2))
            if final_res < 1e-10:
                # Round to half-integers if close
                def round_half(x):
                    y = np.round(x * 2) / 2
                    if np.max(np.abs(x - y)) < 1e-6:
                        return y
                    return x
                Uc = round_half(Uc); Vc = round_half(Vc); Wc = round_half(Wc)
                return {"U": Uc.tolist(), "V": Vc.tolist(), "W": Wc.tolist()}

    # Fallback: return rank-32
    return {"U": U, "V": V, "W": W}
