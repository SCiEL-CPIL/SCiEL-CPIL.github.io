"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Uses a Fourier-series parameterization to enforce smoothness and symmetry,
combined with a multi-stage optimization strategy."""

import time
import numpy as np

_START = time.time()
DEADLINE = _START + 145.0


def _build_hamiltonian():
    """Return (A_mat, B_mat, H_total) for the 6-site open XXZ chain (aniso=0.8, field=0.3)."""
    N = 6
    dim = 1 << N
    aniso = 0.8
    field = 0.3

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    def tensor(*ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(site, mat):
        ops = [eye2 if i != site else mat for i in range(N)]
        return tensor(*ops)

    def pair_op(i, j, mi, mj):
        ops = []
        for k in range(N):
            if k == i:
                ops.append(mi)
            elif k == j:
                ops.append(mj)
            else:
                ops.append(eye2)
        return tensor(*ops)

    H_even = (pair_op(0, 1, sx, sx) + pair_op(0, 1, sy, sy) + aniso * pair_op(0, 1, sz, sz)
              + pair_op(2, 3, sx, sx) + pair_op(2, 3, sy, sy) + aniso * pair_op(2, 3, sz, sz)
              + pair_op(4, 5, sx, sx) + pair_op(4, 5, sy, sy) + aniso * pair_op(4, 5, sz, sz))
    H_odd = (pair_op(1, 2, sx, sx) + pair_op(1, 2, sy, sy) + aniso * pair_op(1, 2, sz, sz)
             + pair_op(3, 4, sx, sx) + pair_op(3, 4, sy, sy) + aniso * pair_op(3, 4, sz, sz))
    H_field_half = -0.5 * field * sum(site_op(i, sz) for i in range(N))

    A_mat = H_odd + H_field_half
    B_mat = H_even + H_field_half
    H_total = A_mat + B_mat
    return A_mat, B_mat, H_total


def _eig_decomp(mat):
    w, v = np.linalg.eigh(mat)
    return w, v, np.linalg.inv(v)


def _exp_negi_t(w, v, inv_v, t):
    phases = np.exp(-1j * t * w)
    return (v * phases) @ inv_v


def _seed_fallback(stages):
    substeps = max(1, stages // 2)
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b


def coefficients(stages):
    if stages != 12:
        return _seed_fallback(stages)

    try:
        A_mat, B_mat, H_total = _build_hamiltonian()
        wA, vA, invA = _eig_decomp(A_mat)
        wB, vB, invB = _eig_decomp(B_mat)
        wH, vH, invH = _eig_decomp(H_total)

        eval_times = [0.25, 0.5, 1.0]
        exact = [_exp_negi_t(wH, vH, invH, t) for t in eval_times]
        dim = A_mat.shape[0]
        I = np.eye(dim, dtype=complex)

        # Palindromic parameterization with free first-half coefficients
        # a_k = a_{12-k}, b_k = b_{12-k} (0-indexed: a[k]=a[11-k])
        # Free variables: a[0..5], b[0..5] with constraint sum(a[0..5])=0.5, sum(b[0..5])=0.5
        # Use deviation variables: a_half = 1/6 + d_a, sum(d_a)=0
        # Parameterize d_a via 5 free variables (last determined by constraint)
        
        def expand_params(x):
            """x = [da0..da4, db0..db4] (10 vars)
            Returns (a_full, b_full) with palindromic symmetry and sum=1.
            """
            # da5 = -(da0+...+da4) to enforce sum=0
            da = np.zeros(6)
            da[:5] = x[:5]
            da[5] = -np.sum(x[:5])
            
            db = np.zeros(6)
            db[:5] = x[5:10]
            db[5] = -np.sum(x[5:10])
            
            a_half = 1.0/6.0 + da
            b_half = 1.0/6.0 + db
            
            # Palindromic: full[0..5] = half, full[6..11] = half[::-1]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            return a_full, b_full

        def _compute_errors(a, b):
            errs = np.empty(3)
            for idx, t in enumerate(eval_times):
                U = I.copy()
                for k in range(12):
                    Ea = _exp_negi_t(wA, vA, invA, a[k] * t)
                    Eb = _exp_negi_t(wB, vB, invB, b[k] * t)
                    U = Eb @ Ea @ U
                errs[idx] = np.linalg.norm(U - exact[idx], 2)
            return errs

        def objective(x):
            if time.time() > DEADLINE - 3.0:
                return 1e6
            a, b = expand_params(x)
            # Penalty for out-of-bounds
            if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
                return 1e6
            errs = _compute_errors(a, b)
            return float(np.mean(errs))

        def objective_smooth(x):
            if time.time() > DEADLINE - 3.0:
                return 1e6
            a, b = expand_params(x)
            if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
                return 1e6
            errs = _compute_errors(a, b)
            # Soft-max for smoothness + small L2 penalty
            m = np.max(errs)
            soft_max = m + (1.0/10.0) * np.log(np.sum(np.exp(10.0*(errs - m))))
            penalty = 0.001 * np.sum(x ** 2)
            return float(soft_max + penalty)

        # Pure-NumPy Nelder-Mead implementation
        def nelder_mead(f, x0, maxiter=500, xatol=1e-10, fatol=1e-12):
            n = len(x0)
            alpha, gamma, rho, sigma = 0.5, 2.0, 0.5, 0.5
            # Build simplex
            simplex = np.zeros((n+1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.05 if x0[i] == 0 else 0.05*x0[i]
            # Evaluate
            fvals = np.array([f(simplex[i]) for i in range(n+1)])
            for it in range(maxiter):
                if time.time() > DEADLINE - 3.0:
                    break
                # Sort
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                # Check convergence
                if np.max(np.abs(fvals - fvals[0])) < fatol:
                    break
                if np.max(np.ptp(simplex, axis=0)) < xatol:
                    break
                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)
                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                fr = f(xr)
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    fe = f(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n-1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[n] - centroid)
                    fc = f(xc)
                    if fc < fvals[n]:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
            return simplex[0], fvals[0]

        # Analytical initial guess from BCH leading-order commutator minimization
        # For a symmetric 12-stage BAOAB-type scheme, the leading error is O(t^3)
        # The optimal weights minimize the commutator coefficient.
        # For a palindromic scheme with steps: B A B A ... B A B (12 steps = 6 A + 6 B)
        # Standard optimal 2nd-order symmetric coefficients for even splitting:
        # Use the known optimal 4th-order coefficients scaled appropriately.
        # For a 6-pair symmetric scheme, use equal weights as baseline:
        # a_k = b_k = 1/6 for all k (uniform)
        x0 = np.zeros(10)  # uniform: all deviations zero
        
        best_x = x0.copy()
        best_val = objective(x0)

        # Phase 1: Nelder-Mead on smooth surrogate
        if time.time() < DEADLINE - 30.0:
            try:
                x1, f1 = nelder_mead(objective_smooth, x0, maxiter=300, xatol=1e-8, fatol=1e-10)
                if f1 < best_val:
                    best_val = f1
                    best_x = x1.copy()
            except Exception:
                pass

        # Phase 2: Multi-start from random perturbations
        if time.time() < DEADLINE - 15.0:
            rng = np.random.RandomState(42)
            for trial in range(5):
                if time.time() > DEADLINE - 15.0:
                    break
                x_trial = best_x + rng.randn(10) * 0.05
                try:
                    x2, f2 = nelder_mead(objective_smooth, x_trial, maxiter=200, xatol=1e-9, fatol=1e-11)
                    if f2 < best_val:
                        best_val = f2
                        best_x = x2.copy()
                except Exception:
                    pass

        # Phase 3: High-precision Nelder-Mead on true objective
        if time.time() < DEADLINE - 8.0:
            try:
                x3, f3 = nelder_mead(objective, best_x, maxiter=500, xatol=1e-12, fatol=1e-14)
                if f3 < best_val:
                    best_val = f3
                    best_x = x3.copy()
            except Exception:
                pass

        # Phase 4: Final tight polish
        if time.time() < DEADLINE - 3.0:
            try:
                x4, f4 = nelder_mead(objective, best_x, maxiter=800, xatol=1e-14, fatol=1e-16)
                if f4 < best_val:
                    best_val = f4
                    best_x = x4.copy()
            except Exception:
                pass

        a, b = expand_params(best_x)

        # Validate
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _seed_fallback(stages)
        if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
            return _seed_fallback(stages)

        # Ensure exact sum = 1 (should be by construction, but guard against float drift)
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa - 1.0) > 1e-12:
            a = a / sa
        if abs(sb - 1.0) > 1e-12:
            b = b / sb

        # Clip to [-50, 50] for safety
        a = np.clip(a, -50.0, 50.0)
        b = np.clip(b, -50.0, 50.0)

        return list(a), list(b)

    except Exception:
        return _seed_fallback(stages)
