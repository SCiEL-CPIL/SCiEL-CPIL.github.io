"""Analytic Hermite-Gaussian uncertainty bound with spherical coordinates + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to (a0, a4, a8).
    a12 is determined by the constraint g(0) = 0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    Key insight: vF = n2/n0 = 7/3 (constant), so bound = sqrt(7/3 * m2/m0).
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    g2 = 2.0 * a0 + 210.0 * a8
    sqrt_pi = np.sqrt(np.pi)

    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    n0 = 18.0 * np.pi * g2 * g2
    n2 = 42.0 * np.pi * g2 * g2

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)

    grad_da0 = 0.5 / bound * vF * dvf_da0
    grad_da4 = 0.5 / bound * vF * dvf_da4
    grad_da8 = 0.5 / bound * vF * dvf_da8

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _bound_and_grad_spherical(theta, phi):
    """
    Spherical coordinate parametrization on the unit sphere.
    a0 = sin(theta)*cos(phi), a4 = sin(theta)*sin(phi), a8 = cos(theta)
    Returns bound and gradient w.r.t. (theta, phi).
    """
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    
    bound, grad_abc = _bound_and_grad(a0, a4, a8)
    
    if not np.isfinite(bound) or bound >= 1e17:
        return bound, np.array([0.0, 0.0])
    
    # Chain rule: d/dtheta and d/dphi
    # da0/dtheta = cos(theta)*cos(phi), da0/dphi = -sin(theta)*sin(phi)
    # da4/dtheta = cos(theta)*sin(phi), da4/dphi = sin(theta)*cos(phi)
    # da8/dtheta = -sin(theta), da8/dphi = 0
    
    grad_theta = (grad_abc[0] * np.cos(theta) * np.cos(phi) +
                  grad_abc[1] * np.cos(theta) * np.sin(phi) +
                  grad_abc[2] * (-np.sin(theta)))
    
    grad_phi = (grad_abc[0] * (-np.sin(theta) * np.sin(phi)) +
                grad_abc[1] * (np.sin(theta) * np.cos(phi)))
    
    return bound, np.array([grad_theta, grad_phi])


def _objective_spherical(params):
    val, _ = _bound_and_grad_spherical(params[0], params[1])
    return val if np.isfinite(val) else 1e18


def _objective_grad_spherical(params):
    val, grad = _bound_and_grad_spherical(params[0], params[1])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0])
    return val, grad


def construct():
    """
    Use spherical coordinates to parametrize the unit sphere in (a0,a4,a8) space.
    This eliminates the scale invariance and reduces the problem to a 2D bounded domain.
    Then use L-BFGS-B with analytic gradients for high-precision optimization.
    """
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    
    # Convert seed to spherical coordinates
    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_unit = seed / norm_seed
        a0_s, a4_s, a8_s = seed_unit
        theta_s = np.arccos(np.clip(a8_s, -1.0, 1.0))
        phi_s = np.arctan2(a4_s, a0_s)
        if phi_s < 0:
            phi_s += 2.0 * np.pi
    else:
        theta_s = np.pi / 2
        phi_s = 0.0
    
    best_val = np.inf
    best_theta, best_phi = theta_s, phi_s
    best_abc = seed.copy()
    
    # Evaluate seed
    val_seed, _ = _bound_and_grad(*seed)
    if np.isfinite(val_seed):
        best_val = val_seed
    
    # Multi-start in spherical coordinates
    rng = np.random.default_rng(42)
    starts = [(theta_s, phi_s)]
    
    # Add structured starts
    for theta in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for phi in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
            starts.append((theta, phi))
    
    # Add random starts
    for _ in range(30):
        starts.append((rng.uniform(0.01, np.pi - 0.01), rng.uniform(0, 2*np.pi)))
    
    # Optimize from each start
    for theta0, phi0 in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad_spherical, np.array([theta0, phi0]),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_theta, best_phi = res.x[0], res.x[1]
                # Convert back to (a0,a4,a8)
                a0_b = np.sin(best_theta) * np.cos(best_phi)
                a4_b = np.sin(best_theta) * np.sin(best_phi)
                a8_b = np.cos(best_theta)
                best_abc = np.array([a0_b, a4_b, a8_b])
        except Exception:
            continue
    
    # Polish in original coordinates from the best spherical solution
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(lambda p: _bound_and_grad(*p)[0], best_abc,
                          method='L-BFGS-B',
                          jac=lambda p: _bound_and_grad(*p)[1],
                          options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_abc = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(lambda p: _bound_and_grad(*p)[0], best_abc,
                          method='L-BFGS-B',
                          jac=lambda p: _bound_and_grad(*p)[1],
                          options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_abc = res.x.copy()
        except Exception:
            pass
    
    # Verify and return
    val_final, _ = _bound_and_grad(*best_abc)
    if np.isfinite(val_final) and val_final < best_val:
        best_val = val_final
    
    return np.array(best_abc)
# EVOLVE-BLOCK-END
