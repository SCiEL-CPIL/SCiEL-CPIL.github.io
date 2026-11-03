"""Analytic uncertainty-inequality tuner.

Structural change from the seed: replace numerical quadrature + generic
optimizers with (a) EXACT integrals via Gaussian-Hermite orthogonality and the
Fourier transform of Hermite-Gaussian functions, and (b) a deterministic
analytic minimization (eigendecomposition of a 3x3 matrix pencil + 1-D
golden-section).  No stochastic search, no coarse grid, no L-BFGS.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import factorial


def _h(n):
    """H_n(0) for even n (physicist's Hermite).  H_{2k}(0) = (-1)^k (2k)!/k!.

    The g(0)=0 constraint uses c12 = -(c0*H0 + c4*H4 + c8*H8)/H12, i.e. the
    ratio c12/c0 = -H0/H12 = -1/40320, c12/c4 = -H4/H12 = -1/3360,
    c12/c8 = -H8/H12 = -1/24.
    """
    return (-1) ** (n // 2) * factorial(n, exact=True) / factorial(n // 2, exact=True)


def _build_system():
    """Return (M, Mx, Mf), each a symmetric 3x3 Gram matrix for the basis
    {v0, v4, v8} where v_{2k} = (H_{2k} - (H_{2k}(0)/H_{12}(0)) H_{12}) e^{-x^2/2}.

    M  = <v_i, v_j>          (L2 norm)
    Mx = <x^2 v_i, v_j>      (position-second-moment)
    Mf = <xi^2 f_i, f_j>     (frequency-second-moment, f_i = F(v_i))
    All integrals are evaluated exactly.
    """
    a = [0, 4, 8]
    H = {n: _h(n) for n in [0, 4, 8, 12]}
    H12 = H[12]
    r = {n: H[n] / H12 for n in [0, 4, 8]}  # r_n = H_n(0)/H12(0)

    # --- M and Mx: Gaussian-Hermite orthogonality ---
    # <H_m H_n e^{-x^2}> = 2^m m! sqrt(pi) delta_{mn}
    # <x^2 H_m H_n e^{-x^2}> = (2m+1) 2^m m! sqrt(pi) delta_{mn}
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    for i, n in enumerate(a):
        Mn = 2.0 ** n * factorial(n, exact=True) * np.sqrt(np.pi)
        M[i, i] = (1.0 + r[n] ** 2) * Mn
        Mx[i, i] = (1.0 + r[n] ** 2) * (2 * n + 1) * Mn
    for i in range(3):
        for j in range(i + 1, 3):
            n, m = a[i], a[j]
            Mnm = 2.0 ** n * factorial(n, exact=True) * np.sqrt(np.pi)  # n<m so use n
            M[i, j] = M[j, i] = (r[n] + r[m]) * Mnm
            Mx[i, j] = Mx[j, i] = ((2 * n + 1) * r[n] + (2 * m + 1) * r[m]) * Mnm

    # --- Mf: Fourier transform of H_n e^{-x^2/2} ---
    # F{H_n e^{-x^2/2}}(xi) = (-i)^n H_n(xi) e^{-xi^2/2}  (unitary convention)
    # so |f_n|^2 = |f_m|^2 = e^{-xi^2}, and
    # <xi^2 f_i, f_j> = (1 + r_n r_m) * (2n+1) 2^n n! sqrt(pi)   (i==j)
    #                    (r_n r_m)(2n+1) 2^n n! sqrt(pi)          (i != j)
    Mf = np.zeros((3, 3))
    for i, n in enumerate(a):
        Mn = 2.0 ** n * factorial(n, exact=True) * np.sqrt(np.pi)
        Mf[i, i] = (1.0 + r[n] ** 2) * (2 * n + 1) * Mn
    for i in range(3):
        for j in range(i + 1, 3):
            n, m = a[i], a[j]
            Mnm = 2.0 ** n * factorial(n, exact=True) * np.sqrt(np.pi)
            Mf[i, j] = Mf[j, i] = r[n] * r[m] * (2 * n + 1) * Mnm

    return M, Mx, Mf


def _bound(c, M, Mx, Mf):
    """Upper bound C4 = sqrt(<x^2 g^2> <xi^2 |F g|^2>) / <g^2> for g = c.v."""
    c = np.asarray(c, dtype=float)
    norm2 = c @ M @ c
    if norm2 <= 0.0 or not np.isfinite(norm2):
        return np.inf
    x2 = c @ Mx @ c
    f2 = c @ Mf @ c
    if x2 < 0.0 or f2 < 0.0 or not np.isfinite(x2) or not np.isfinite(f2):
        return np.inf
    return np.sqrt(x2 * f2) / norm2


def _golden_section_1d(f, lo, hi, iters=220):
    """Minimize a 1-D function on [lo, hi] by golden-section search."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def construct():
    start_time = time.time()
    deadline = 180.0  # evaluator kills at 210s; keep a safety margin

    M, Mx, Mf = _build_system()

    # --- Deterministic analytic minimization over direction of c in R^3 ---
    # Solve the generalized eigenproblem (Mx, M):  Mx u = lambda M u.
    # The two eigenvalues lambda_1 > lambda_2 bracket the achievable
    # x2/norm2.  Along the plane spanned by the two eigenvectors,
    # c = sqrt(l1+l2) cos(theta) u1 + sqrt(l1-l2) sin(theta) u2, and the
    # bound becomes a 1-D function of theta minimized by golden-section.
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ Mx)
        idx = np.argsort(eigvals)[::-1]
        l1 = float(eigvals[idx[0]])
        l2 = float(eigvals[idx[1]])
        u1 = eigvecs[:, idx[0]].real
        u2 = eigvecs[:, idx[1]].real

        # orthonormalize u2 against u1 (numerical safety)
        u2 = u2 - np.dot(u2, u1) * u1
        n2 = np.linalg.norm(u2)
        if n2 < 1e-12:
            u2 = np.array([0.0, 0.0, 1.0]) - np.dot([0, 0, 1], u1) * u1
            u2 = u2 / np.linalg.norm(u2)
        else:
            u2 = u2 / n2

        s1 = np.sqrt(l1 + l2)
        s2 = np.sqrt(max(l1 - l2, 0.0))

        def f1d(theta):
            c = s1 * np.cos(theta) * u1 + s2 * np.sin(theta) * u2
            return _bound(c, M, Mx, Mf)

        theta_star = _golden_section_1d(f1d, 0.0, np.pi, iters=240)
        c_best = s1 * np.cos(theta_star) * u1 + s2 * np.sin(theta_star) * u2
        best = _bound(c_best, M, Mx, Mf)
    except Exception:
        c_best = np.array([1.0, 0.0, 0.0])
        best = _bound(c_best, M, Mx, Mf)

    # --- Deterministic multi-start refinement (cheap, exact, bounded) ---
    if time.time() - start_time < deadline - 3.0:
        # Candidate directions: eigenvectors of (Mf, M) and of (Mx, Mf),
        # plus the analytic solution, plus a few deterministic spherical samples.
        cand_dirs = [c_best]
        for A in (Mf, Mx):
            try:
                _, ev = np.linalg.eig(np.linalg.inv(M) @ A)
                for v in ev:
                    v = v.real
                    if np.linalg.norm(v) > 1e-10:
                        cand_dirs.append(v / np.linalg.norm(v))
            except Exception:
                pass
        # deterministic spherical samples (theta, phi) on a small fixed set
        for t in (0.5, 1.0, 1.5, 2.0, 2.5):
            for p in (0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6):
                cand_dirs.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

        for d in cand_dirs:
            d = np.asarray(d, dtype=float)
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                continue
            d = d / nd
            # local 1-D refinement along d using the (Mx,M) eigendecomposition
            # is already covered; here just evaluate and keep best direction.
            b = _bound(d, M, Mx, Mf)
            if np.isfinite(b) and b < best:
                best = b
                c_best = d
        # final golden-section polish on the best direction's theta (recompute)
        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ Mx)
            idx = np.argsort(eigvals)[::-1]
            l1 = float(eigvals[idx[0]]); l2 = float(eigvals[idx[1]])
            u1 = eigvecs[:, idx[0]].real
            u2 = eigvecs[:, idx[1]].real
            u2 = u2 - np.dot(u2, u1) * u1
            u2 = u2 / np.linalg.norm(u2)
            s1 = np.sqrt(l1 + l2); s2 = np.sqrt(max(l1 - l2, 0.0))
            # seed near c_best by projecting to find a good theta
            def f1d(theta):
                c = s1 * np.cos(theta) * u1 + s2 * np.sin(theta) * u2
                return _bound(c, M, Mx, Mf)
            theta_star = _golden_section_1d(f1d, 0.0, np.pi, iters=260)
            c2 = s1 * np.cos(theta_star) * u1 + s2 * np.sin(theta_star) * u2
            b2 = _bound(c2, M, Mx, Mf)
            if np.isfinite(b2) and b2 < best:
                best = b2
                c_best = c2
        except Exception:
            pass

    c_best = np.asarray(c_best, dtype=float)
    # return the three free coefficients (degrees 0, 4, 8)
    return c_best
# EVOLVE-BLOCK-END
