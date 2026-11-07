"""Suite task: uncertainty inequality. Minimize the largest real root z* of the
degree-5 polynomial q(z) = g(z)/z (z = x^2); value = z*/(2*pi), lower is better."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Fixed exact values at 0 (physicists' Hermite polynomials):
# H0(0)=1, H4(0)=12, H8(0)=1008, H12(0)=103950
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1008.0
H12_0 = 103950.0

# q(z) = c0 + 12 c1 z + 1008 c2 z^2 - 103950 c3 z^3
# with c3 = -(c0 + 12 c1 + 1008 c2)/103950
# => q(z) = c0 + 12 c1 z + 1008 c2 z^2 + (c0 + 12 c1 + 1008 c2) z^3
# Coefficients (constant .. z^3):
K1 = 12.0
K2 = 1008.0
K3 = 1.0


def _q_coeffs(c):
    c0, c1, c2 = c
    s = c0 + K1 * c1 + K2 * c2
    return np.array([c0, K1 * c1, K2 * c2, s])


def _largest_real_root(coeffs):
    """Largest real root of a real polynomial (coeffs[0]=constant .. coeffs[-1]=lead).
    Returns +inf if the leading coefficient is non-positive (invalid) or no real root."""
    lead = coeffs[-1]
    if lead <= 0.0:
        return math.inf
    roots = np.roots(coeffs)
    reals = roots[np.abs(roots.imag) < 1e-8 * np.maximum(1.0, np.abs(roots.real))].real
    if reals.size == 0:
        return math.inf
    return float(np.max(reals))


def _objective(c):
    z = _largest_real_root(_q_coeffs(c))
    return z


# --- Multi-start Nelder-Mead with a global restart strategy -----------------
def _nm_minimize(f, x0, max_iter=300, tol=1e-11, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    fvals = [f(simplex[0])]
    # Build initial simplex
    for i in range(n):
        pt = np.array(x0, dtype=float)
        step = 0.25 + 0.1 * i
        pt[i] += step
        simplex.append(pt)
        fvals.append(f(pt))
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    best_i = int(np.argmin(fvals))
    return simplex[best_i], fvals[best_i]


def _search(deadline):
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_c = seed.copy()
    best_z = _objective(seed)

    rng = np.random.default_rng(20240517)
    starts = [seed]
    # A handful of structured + random starts around the seed
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        for _ in range(3):
            starts.append(seed + scale * rng.standard_normal(3))
    # deterministic grid of small perturbations
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for b in (-1.0, -0.5, 0.0, 0.5, 1.0):
            starts.append(seed + np.array([a, b, 0.0]))

    for s in starts:
        if time.time() > deadline - 25.0:
            break
        c, z = _nm_minimize(_objective, s, max_iter=400, tol=1e-12)
        if z < best_z:
            best_z = z
            best_c = c
        if time.time() > deadline - 15.0:
            break

    # High-precision polish pass: tight-tolerance Nelder-Mead from the best point
    if time.time() < deadline - 8.0:
        c, z = _nm_minimize(_objective, best_c, max_iter=2000, tol=1e-14)
        if z < best_z:
            best_z = z
            best_c = c
    return best_c, best_z


def construct():
    start = time.time()
    deadline = start + 200.0  # safety margin vs 210 s wall-clock kill
    try:
        c, z = _search(deadline)
    except Exception:
        c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    # Final guard: if we are still over budget or something is off, fall back to seed
    if time.time() > deadline - 3.0 or not np.all(np.isfinite(c)):
        c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return np.asarray(c, dtype=float)
# EVOLVE-BLOCK-END
