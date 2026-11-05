"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Uses a Fourier-series parameterization to enforce smoothness and symmetry,
combined with a multi-stage optimization strategy."""

import time
import numpy as np

_START = time.time()
DEADLINE = _START + 130.0


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

        # Fourier-series parameterization for palindromic symmetry
        # a_k = a_{13-k}, b_k = b_{13-k}
        # Parameterize the first 6 coefficients using a truncated Fourier series
        # that naturally produces smooth, symmetric distributions
        #
        # For k = 0..5 (first half), use:
        #   a_half[k] = 1/6 + sum_{m=1}^{M} (c_m * cos(2*pi*m*k/6) + s_m * sin(2*pi*m*k/6))
        # The full 12-coefficient array is then constructed by palindromic extension.
        #
        # With M=2, we have 4 free parameters for a and 4 for b (total 8).
        # The mean is fixed at 1/6 for each half, so sum of first half = 0.5,
        # and by palindromic symmetry, sum of full array = 1.

        M = 2  # number of Fourier modes

        def fourier_expand(x):
            """Expand 2*(M+1) Fourier coefficients to 12 palindromic coefficients.
            x = [a_c1, a_s1, a_c2, a_s2, b_c1, b_s1, b_c2, b_s2]
            """
            k_idx = np.arange(6)  # 0..5 for first half
            theta = 2.0 * np.pi * k_idx / 6.0

            # Build first half of a
            a_half = np.full(6, 1.0 / 6.0)
            for m in range(1, M + 1):
                a_half += x[2 * (m - 1)] * np.cos(m * theta)
                a_half += x[2 * (m - 1) + 1] * np.sin(m * theta)

            # Build first half of b
            b_half = np.full(6, 1.0 / 6.0)
            for m in range(1, M + 1):
                b_half += x[2 * (m - 1) + 2 * M] * np.cos(m * theta)
                b_half += x[2 * (m - 1) + 2 * M + 1] * np.sin(m * theta)

            # Palindromic extension: full[k] = half[k] for k=0..5, full[11-k] = half[k]
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

        def objective_fourier(x):
            if time.time() > DEADLINE - 3.0:
                return 1e6
            a, b = fourier_expand(x)
            errs = _compute_errors(a, b)
            return float(np.mean(errs))

        def objective_smooth_fourier(x):
            if time.time() > DEADLINE - 3.0:
                return 1e6
            a, b = fourier_expand(x)
            errs = _compute_errors(a, b)
            beta = 15.0
            m = np.max(errs)
            soft_max = m + np.log(np.sum(np.exp(errs - m)))
            penalty = 0.005 * np.sum(x ** 2)
            return float(soft_max + penalty)

        # Initial guess: uniform (all Fourier coefficients zero)
        x0 = np.zeros(2 * (M + 1))

        best_x = x0.copy()
        best_val = objective_fourier(x0)

        # Phase 1: Nelder-Mead on smooth surrogate
        from scipy.optimize import minimize
        if time.time() < DEADLINE - 25.0:
            try:
                res = minimize(objective_smooth_fourier, x0, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Differential evolution for global exploration
        if time.time() < DEADLINE - 15.0:
            try:
                from scipy.optimize import differential_evolution
                bounds = [(-0.15, 0.15)] * (2 * (M + 1))
                res2 = differential_evolution(
                    objective_smooth_fourier, bounds, seed=42,
                    maxiter=80, tol=1e-9, popsize=15,
                    mutation=(0.4, 1.0), recombination=0.8,
                    polish=True,
                    options={'init': 'sobol', 'maxiter': 80}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        # Phase 3: High-precision Nelder-Mead polish on true objective
        if time.time() < DEADLINE - 5.0:
            try:
                res3 = minimize(objective_fourier, best_x, method='Nelder-Mead',
                                options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-15,
                                         'adaptive': True})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_x = res3.x.copy()
            except Exception:
                pass

        # Phase 4: Final tight polish
        if time.time() < DEADLINE - 2.0:
            try:
                res4 = minimize(objective_fourier, best_x, method='Nelder-Mead',
                                options={'maxiter': 800, 'xatol': 1e-15, 'fatol': 1e-16,
                                         'adaptive': True})
                if res4.fun < best_val:
                    best_val = res4.fun
                    best_x = res4.x.copy()
            except Exception:
                pass

        a, b = fourier_expand(best_x)

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

        return list(a), list(b)

    except Exception:
        return _seed_fallback(stages)
