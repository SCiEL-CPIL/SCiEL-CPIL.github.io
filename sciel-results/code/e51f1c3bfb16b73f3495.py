"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 82.0  # safety margin below 90s kill
_T0 = time.time()


def _E_of_z(z, w0, wa):
    """CPL E(z) = (1+z)^(3(1+w0+wa)) * (1+z)^(-3wa) = (1+z)^(3+3w0) * (1+z)^(-3wa)."""
    zp = 1.0 + z
    return zp ** (3.0 + 3.0 * w0 - 3.0 * wa)


def _true_luminosity_distance(z, w0, wa, n=40000):
    """High-precision reference via adaptive Simpson (n uniform subintervals)."""
    if z <= 0.0:
        return 0.0
    # dL/(c/H0) = (1+z) * integral_0^z dz' / E(z')
    zz = np.linspace(0.0, z, n)
    ee = _E_of_z(zz, w0, wa)
    integrand = 1.0 / ee
    # composite Simpson
    h = z / (n - 1)
    s = integrand[0] + integrand[-1]
    s += 4.0 * np.sum(integrand[1:-1:2])
    s += 2.0 * np.sum(integrand[2:-1:2])
    integral = (h / 3.0) * s
    return (1.0 + z) * integral


def _approx_luminosity_distance(z, nodes, weights, w0, wa):
    """Quadrature estimate: (1+z)*z*sum_i weights[i]/E(a_i) with a_i=1/(1+z*nodes[i])."""
    if z <= 0.0:
        return 0.0
    a = 1.0 / (1.0 + z * nodes)
    # E(a) in terms of a: E(a)=a^(-3(1+w0+wa)) * a^(3wa) = a^(-3-3w0+3wa)
    e = a ** (-3.0 - 3.0 * w0 + 3.0 * wa)
    return (1.0 + z) * z * np.sum(weights / e)


def _rel_errs(nodes, weights, w0s, was, zs):
    """Vectorized relative errors over models and redshifts (flattened)."""
    errs = []
    for w0, wa in zip(w0s, was):
        for z in zs:
            true_val = _true_luminosity_distance(z, w0, wa)
            approx = _approx_luminosity_distance(z, nodes, weights, w0, wa)
            if true_val > 0:
                errs.append(abs(approx - true_val) / true_val)
    return np.array(errs)


def _soft_max(errs, beta=200.0):
    """Smooth max via log-sum-exp."""
    m = np.max(errs)
    return m + np.log(np.sum(np.exp(beta * (errs - m)))) / beta


def _objective(theta, w0s, was, zs, beta):
    # theta: [x_0..x_6 (node log-spacings), w_0..w_5 (weight logits)]
    n = 8
    x = theta[:7]
    wlogits = theta[7:]
    # Nodes: start from Gauss-Legendre base, add small monotone perturbation via cumulative softplus
    base = np.array(np.polynomial.legendre.leggauss(n)[0])
    # x as perturbations of log-spacings; ensure monotone increasing
    # Use cumulative sum of softplus(x) + small epsilon to get strictly increasing increments
    eps = 1e-6
    inc = np.softplus(x) + eps
    cum = np.cumsum(inc)
    nodes = base + 0.01 * (cum - cum[0])
    # clip to (0,1) while preserving strict increase
    nodes = np.clip(nodes, 1e-4, 1.0 - 1e-4)
    # enforce strict increase
    for i in range(1, n):
        if nodes[i] <= nodes[i - 1]:
            nodes[i] = nodes[i - 1] + 1e-6
    # Weights via softmax
    wexp = np.exp(wlogits - np.max(wlogits))
    weights = wexp / np.sum(wexp)
    # sum-to-1 check
    weights = weights / np.sum(weights)
    errs = _rel_errs(nodes, weights, w0s, was, zs)
    return _soft_max(errs, beta), nodes, weights


def construct():
    global _T0
    # Deadline guard
    def remaining():
        return _DEADLINE - (time.time() - _T0)

    if remaining() < 1.0:
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    # Diverse CPL model set (w0, wa)
    w0s = [-1.2, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, 0.0, 0.2, 0.4]
    was = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    # Redshift grid for evaluation
    zs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    # Initial theta from midpoint rule
    n = 8
    base = np.array(np.polynomial.legendre.leggauss(n)[0])
    x0 = np.zeros(7)
    wlogits0 = np.zeros(7)
    theta0 = np.concatenate([x0, wlogits0])

    beta = 200.0
    best_nodes = (np.arange(n) + 0.5) / n
    best_weights = np.full(n, 1.0 / n)
    best_obj = np.inf

    def obj_only(theta):
        return _objective(theta, w0s, was, zs, beta)[0]

    try:
        res = minimize(
            obj_only,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50},
        )
        if res.success or res.fun < best_obj:
            _, nodes, weights = _objective(res.x, w0s, was, zs, beta)
            best_nodes, best_weights, best_obj = nodes, weights, res.fun
    except Exception:
        pass

    if remaining() < 2.0:
        return {"nodes": best_nodes, "weights": best_weights}

    # Polish pass with higher beta and tighter tolerance
    beta2 = 500.0
    try:
        def obj_only2(theta):
            return _objective(theta, w0s, was, zs, beta2)[0]

        res2 = minimize(
            obj_only2,
            res.x,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-12, "maxls": 50},
        )
        if res2.fun < best_obj:
            _, nodes, weights = _objective(res2.x, w0s, was, zs, beta2)
            best_nodes, best_weights, best_obj = nodes, weights, res2.fun
    except Exception:
        pass

    # Ensure nodes strictly in (0,1) and strictly increasing
    best_nodes = np.clip(best_nodes, 1e-4, 1.0 - 1e-4)
    for i in range(1, len(best_nodes)):
        if best_nodes[i] <= best_nodes[i - 1]:
            best_nodes[i] = best_nodes[i - 1] + 1e-6
    # Weights positive and sum to 1
    best_weights = np.abs(best_weights)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
