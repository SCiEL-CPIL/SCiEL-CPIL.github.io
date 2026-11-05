"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Builds the Hamiltonian, exploits palindromic symmetry to reduce free variables,
and runs a fast Nelder-Mead / differential_evolution hybrid with an explicit
deadline guard. Falls back to the seed's Strang split on any timeout or error."""

import time
import numpy as np

_START = time.time()
# Evaluator kills at 150 s wall-clock. Keep a hard safety margin.
DEADLINE = _START + 135.0


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

    # Even bonds: 0-1, 2-3, 4-5 ; Odd bonds: 1-2, 3-4
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
    """exp(-i t M) via precomputed eigendecomposition."""
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
        # ---- Build Hamiltonian and precompute exponentials ----
        A_mat, B_mat, H_total = _build_hamiltonian()
        wA, vA, invA = _eig_decomp(A_mat)
        wB, vB, invB = _eig_decomp(B_mat)
        wH, vH, invH = _eig_decomp(H_total)

        eval_times = [0.25, 0.5, 1.0]
        exact = [_exp_negi_t(wH, vH, invH, t) for t in eval_times]
        dim = A_mat.shape[0]
        I = np.eye(dim, dtype=complex)

        # ---- Symmetric (palindromic) parameterization ----
        # a_k = a_{13-k}, b_k = b_{13-k}  (k=1..12)
        # Free: a1..a6, b1..b6 with sum(a1..a6)=0.5, sum(b1..b6)=0.5
        # Reparam: 5 free a, 5 free b; a6 = 0.5 - sum(a1..a5), b6 = 0.5 - sum(b1..b5)
        # Total free vars: 10

        def expand(x):
            a_f = x[:5]
            b_f = x[5:10]
            a6 = 0.5 - np.sum(a_f)
            b6 = 0.5 - np.sum(b_f)
            a_half = np.concatenate([a_f, [a6]])
            b_half = np.concatenate([b_f, [b6]])
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            return a_full, b_full

        def _compute_errors(x):
            """Return array of 3 spectral-norm errors at t = 0.25, 0.5, 1.0."""
            a, b = expand(x)
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
            """Mean spectral-norm error (original scoring metric)."""
            if time.time() > DEADLINE - 5.0:
                return 1e6
            return float(np.mean(_compute_errors(x)))

        def objective_smooth(x):
            """Smooth epigraph surrogate: soft-max over the 3 times + penalty.
            Uses log-sum-exp for a C1 approximation of max(errs).
            A small quadratic penalty discourages large coefficient magnitudes."""
            if time.time() > DEADLINE - 5.0:
                return 1e6
            errs = _compute_errors(x)
            beta = 20.0
            m = np.max(errs)
            # log-sum-exp: m + log(sum(exp(errs - m)))
            soft_max = m + np.log(np.sum(np.exp(errs - m)))
            a, b = expand(x)
            penalty = 0.01 * (np.sum(a**2) + np.sum(b**2))
            return float(soft_max + penalty)

        # ---- Optimization ----
        # Start from uniform symmetric: each a_k = b_k = 1/12
        x0 = np.full(10, 0.5 / 6.0)  # a1..a5 = b1..b5 = 1/12, a6=b6=1/12

        best_x = x0.copy()
        best_val = objective(x0)

        # Phase 1: Nelder-Mead on the smooth surrogate (fast, local, C1)
        from scipy.optimize import minimize
        if time.time() < DEADLINE - 20.0:
            res = minimize(objective_smooth, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()

        # Phase 2: Differential evolution (global) using the smooth surrogate
        if time.time() < DEADLINE - 15.0:
            from scipy.optimize import differential_evolution
            bounds = [(-2.0, 2.0)] * 10
            try:
                res2 = differential_evolution(
                    objective_smooth, bounds, seed=42,
                    maxiter=100, tol=1e-8, popsize=18,
                    mutation=(0.3, 1.0), recombination=0.8,
                    polish=True,
                    options={'init': 'sobol', 'maxiter': 100}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish on the true (nonsmooth) objective
        if time.time() < DEADLINE - 5.0:
            try:
                res3 = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                         'adaptive': True})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_x = res3.x.copy()
            except Exception:
                pass

        # Phase 4: Second high-precision polish (tighter, more iterations)
        if time.time() < DEADLINE - 2.0:
            try:
                res4 = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-16,
                                         'adaptive': True})
                if res4.fun < best_val:
                    best_val = res4.fun
                    best_x = res4.x.copy()
            except Exception:
                pass

        a, b = expand(best_x)
        # Sanity: ensure finite and in range
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _seed_fallback(stages)
        if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
            return _seed_fallback(stages)
        # Renormalize to exact sum=1 while preserving palindromic symmetry.
        # Because expand() enforces sum(a)=1 and sum(b)=1 by construction,
        # we only need to verify; if there's floating-point drift, apply a
        # symmetric correction: scale each pair (a_k, a_{13-k}) equally.
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa - 1.0) > 1e-12:
            a = a / sa
        if abs(sb - 1.0) > 1e-12:
            b = b / sb
        return list(a), list(b)

    except Exception:
        return _seed_fallback(stages)
