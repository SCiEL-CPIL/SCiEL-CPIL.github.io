import numpy as np
from scipy import optimize
import time


def _variance_product(coeffs):
    """Compute variance of g * variance of F(g) for the Hermite-Gaussian test function."""
    a0, a4, a8 = coeffs
    # Enforce g(0) = 0: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=10080
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 10080.0

    x = np.linspace(-8.0, 8.0, 16000)
    x2 = x * x

    # Physicists' Hermite polynomials
    H0 = np.ones_like(x)
    H4 = 16.0 * x2**2 - 48.0 * x2 + 12.0
    H8 = 256.0 * x2**4 - 3072.0 * x2**3 + 11520.0 * x2**2 - 13440.0 * x2 + 1680.0
    H12 = (4096.0 * x2**6 - 65536.0 * x2**5 + 368640.0 * x2**4
           - 745440.0 * x2**3 + 498960.0 * x2**2 - 60480.0 * x2 + 10080.0)

    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = P * np.exp(-0.5 * x2)

    norm2 = np.trapz(g * g, x)
    if norm2 <= 1e-30:
        return 1e10

    w = g * g / norm2  # normalized weight
    x_mean = np.trapz(x * w, x)
    x2_mean = np.trapz(x2 * w, x)
    var = x2_mean - x_mean * x_mean

    # For even-degree Hermite polynomials, the Fourier transform preserves the
    # polynomial structure (i^n = 1 for even n), so var_F(g) = var_g.
    # The product is var^2.
    return var * var


def construct():
    t0 = time.time()
    deadline = t0 + 170.0  # safety margin before 180s kill

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _variance_product(seed)

    # Primary local optimization from seed
    try:
        res = optimize.minimize(
            _variance_product, seed, method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if time.time() < deadline and res.fun < best_val:
            best_val = res.fun
            best = res.x.copy()
    except Exception:
        pass

    # Random restarts with local refinement if time permits
    if time.time() < deadline - 40.0:
        rng = np.random.default_rng(42)
        scales = np.array([1.0, 0.1, 0.01])
        for _ in range(300):
            if time.time() > deadline - 30.0:
                break
            cand = seed + rng.standard_normal(3) * scales
            v = _variance_product(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
                # Local polish
                try:
                    res2 = optimize.minimize(
                        _variance_product, best, method='Nelder-Mead',
                        options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best = res2.x.copy()
                except Exception:
                    pass

    return best
