"""Analytic uncertainty bound with spherical reparameterization + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    # Use recurrence: I_n = (2n-1) * I_{n-1} / (4*n) for n>=1, I_0 = sqrt(pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Spherical reparameterization:
# The g(0)=0 constraint means a0 + 3*a4 + 105*a8 = 0 (up to the H12 term).
# We work in the 3D space of (a0, a4, a8) but the objective is scale-invariant 
# (multiplying all coefficients by c doesn't change the uncertainty product).
# So we fix the norm and search on the 2D sphere.
# 
# Let v = (a0, a4, a8). The constraint is a0 + 3*a4 + 105*a8 = 0.
# This defines a 2D plane in R^3. We parametrize points on this plane with 
# two angles.
# 
# A basis for the plane a0 + 3*a4 + 105*a8 = 0:
#   e1 = (1, -1/3, 0)  -> normalize
#   e2 = (105, 0, -1)  -> normalize
# 
# Actually, let's use a cleaner basis:
#   u1 = (3, -1, 0)   (satisfies 3*3 + (-1)*(-1) + 0*105 = 9+1 = 10... no)
# Let me redo: constraint is a0 + 3*a4 + 105*a8 = 0
#   u1 = (3, -1, 0): 3*3 + 3*(-1) + 105*0 = 9-3 = 6 ≠ 0
# 
# Let me just use: 
#   u1 = (1, -1/3, 0): 1 + 3*(-1/3) + 0 = 1-1 = 0 ✓
#   u2 = (105, 0, -1): 105 + 0 + 105*(-1) = 0 ✓
# 
# Normalize:
#   ||u1|| = sqrt(1 + 1/9) = sqrt(10/9) = sqrt(10)/3
#   ||u2|| = sqrt(11025 + 1) = sqrt(11026)
# 
# Orthonormal basis:
#   n1 = u1 / ||u1|| = (3/sqrt(10), -1/sqrt(10), 0)
#   n2 = u2 / ||u2|| = (105/sqrt(11026), 0, -1/sqrt(11026))
# 
# Any point on the plane: v = r * (cos(theta) * n1 + sin(theta) * n2)
# where r > 0 is the scale and theta ∈ [0, 2π).
# Since the objective is scale-invariant, we fix r=1 and search over theta.
# But wait - we need 2D because the plane is 2D. Let me reconsider.
# 
# Actually the constraint a0 + 3*a4 + 105*a8 = 0 gives a 2D subspace.
# A point in this subspace: v = s1 * w1 + s2 * w2 where w1, w2 span the plane.
# The scale invariance means (s1, s2) and c*(s1, s2) give the same answer.
# So we can fix s1^2 + s2^2 = 1 and search over angle phi: s1 = cos(phi), s2 = sin(phi).
# This gives a 1D search! But the directive says 2D spherical coordinates.
# 
# Wait - maybe I'm wrong about scale invariance. Let me check:
# g(x) = P(x) * exp(-x^2/2) where P is the polynomial.
# If we scale P -> c*P, then g -> c*g, so |g|^2 -> c^2*|g|^2, and 
# integral |g|^2 -> c^2 * integral |g|^2.
# Similarly for Fourier. So A(f)*A(F) is scale-invariant. Yes!
# 
# So actually we only need 1D: the angle phi in the 2D constraint plane.
# But let me keep it as 2D for generality and to match the directive.
# Actually, since it's truly 1D after fixing scale, let me just use phi.
# 
# Hmm, but the directive explicitly says "2D spherical coordinate system on the unit sphere".
# Let me re-read: "Parametrize the Hermite coefficients (a0, a4, a8) via a 2D spherical 
# coordinate system on the unit sphere defined by the g(0)=0 constraint"
# 
# I think they mean: the constraint defines a 2D plane, and we use spherical coords 
# on the unit circle in that plane (which is 1D angular parameter). Or perhaps they're 
# thinking of the full 3D sphere intersected with the plane giving a circle.
# 
# Let me just implement it as a 1D angular search since that's mathematically correct,
# but I'll structure it to be extensible. Actually, for the optimization to work well,
# let me use 2 parameters: one for the angle in the constraint plane, and one for... 
# 
# Wait, I think I need to reconsider. The three coefficients are (a0, a4, a8). The 
# constraint g(0)=0 fixes a12 in terms of (a0, a4, a8), but (a0, a4, a8) themselves 
# are free (3 parameters). The constraint a0 + 3*a4 + 105*a8 + 945*a12 = 0 is 
# automatically satisfied by the definition of a12. So (a0, a4, a8) are 3 free 
# parameters, and the objective is scale-invariant, reducing to 2D.
# 
# So: 3 free params - 1 scale invariance = 2D search space.
# We can parametrize as: v = r * (cos(theta)*n1 + sin(theta)*cos(phi)*n2 + sin(phi)*n3)
# where n1, n2, n3 are orthonormal basis of R^3, and we fix r=1.
# But that's the full sphere, not constrained.
# 
# Actually, I realize the constraint is just on a12, not on (a0,a4,a8). So (a0,a4,a8) 
# are genuinely 3 free parameters, and scale invariance reduces to 2D.
# 
# Let me use spherical coordinates on the unit sphere in R^3:
#   a0 = cos(theta)
#   a4 = sin(theta) * cos(phi)  
#   a8 = sin(theta) * sin(phi)
# where theta ∈ [0, pi], phi ∈ [0, 2pi).
# This covers all directions in R^3, and since the objective is scale-invariant,
# fixing the norm to 1 is without loss of generality.
# 
# This is the "2D spherical coordinate system on the unit sphere" mentioned in the directive.

def _spherical_to_coefficients(theta, phi):
    """Convert spherical coordinates to (a0, a4, a8) on unit sphere."""
    a0 = np.cos(theta)
    a4 = np.sin(theta) * np.cos(phi)
    a8 = np.sin(theta) * np.sin(phi)
    return np.array([a0, a4, a8])

def _coefficients_to_spherical(a0, a4, a8):
    """Convert (a0, a4, a8) to (theta, phi) on unit sphere."""
    r = np.sqrt(a0**2 + a4**2 + a8**2)
    if r < 1e-15:
        return (np.pi/2, 0.0)
    a0n, a4n, a8n = a0/r, a4/r, a8/r
    theta = np.arccos(np.clip(a0n, -1.0, 1.0))
    phi = np.arctan2(a8n, a4n)
    return (theta, phi)


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    dpda = np.zeros((3, 7))
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _bound_from_spherical(theta, phi):
    """Compute bound from spherical coordinates (theta, phi) on unit sphere."""
    coeffs = _spherical_to_coefficients(theta, phi)
    val, _ = _bound_and_grad(coeffs)
    return val


def _bound_and_grad_spherical(params):
    """
    Compute bound and gradient in spherical coordinates (theta, phi).
    Uses chain rule: d/d(theta,phi) = (d/d a0, d/d a4, d/d a8) @ (d(a0,a4,a8)/d(theta,phi))
    """
    theta, phi = params[0], params[1]
    coeffs = _spherical_to_coefficients(theta, phi)
    bound, grad_coeff = _bound_and_grad(coeffs)
    
    # Jacobian of (a0, a4, a8) w.r.t. (theta, phi)
    # a0 = cos(theta), a4 = sin(theta)*cos(phi), a8 = sin(theta)*sin(phi)
    # da0/dtheta = -sin(theta), da0/dphi = 0
    # da4/dtheta = cos(theta)*cos(phi), da4/dphi = -sin(theta)*sin(phi)
    # da8/dtheta = cos(theta)*sin(phi), da8/dphi = sin(theta)*cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    jac = np.array([
        [-st, 0.0],
        [ct*cp, -st*sp],
        [ct*sp, st*cp]
    ])
    
    grad_spherical = jac.T @ grad_coeff
    return bound, grad_spherical


def _objective_spherical(params):
    val, _ = _bound_and_grad_spherical(params)
    return val


def _objective_grad_spherical(params):
    val, grad = _bound_and_grad_spherical(params)
    return val, grad


def construct():
    # Convert seed to spherical coordinates
    seed_coeffs = np.array([-1.13, 0.04, 0.000308642])
    seed_theta, seed_phi = _coefficients_to_spherical(*seed_coeffs)
    
    # Generate diverse starting points in spherical coordinates
    rng = np.random.default_rng(42)
    starts = []
    
    # Seed point
    starts.append(np.array([seed_theta, seed_phi]))
    
    # Grid of starting points on the sphere
    for theta in np.linspace(0.1, np.pi - 0.1, 20):
        for phi in np.linspace(0.1, 2*np.pi - 0.1, 20):
            starts.append(np.array([theta, phi]))
    
    # Random points
    for _ in range(50):
        theta = rng.uniform(0.05, np.pi - 0.05)
        phi = rng.uniform(0.05, 2*np.pi - 0.05)
        starts.append(np.array([theta, phi]))
    
    best_val = np.inf
    best_params_sph = None
    best_params_linear = None
    
    # Phase 1: Multi-start L-BFGS-B in spherical coordinates
    for start in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            res = minimize(_objective_grad_spherical, start, method='L-BFGS-B',
                          jac=True, bounds=[(0.01, np.pi - 0.01), (0.01, 2*np.pi - 0.01)],
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 30})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_sph = res.x.copy()
                best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Phase 2: Refine in linear coordinates from best spherical result
    if best_params_linear is not None and time.time() < _DEADLINE - 50:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish in linear coordinates
    if best_params_linear is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective, best_params_linear, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Convert back to spherical and do high-precision polish there
    if best_params_linear is not None and time.time() < _DEADLINE - 15:
        theta, phi = _coefficients_to_spherical(*best_params_linear)
        try:
            res = minimize(_objective_grad_spherical, np.array([theta, phi]), method='L-BFGS-B',
                          jac=True, bounds=[(0.001, np.pi - 0.001), (0.001, 2*np.pi - 0.001)],
                          options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_sph = res.x.copy()
                best_params_linear = _spherical_to_coefficients(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise Nelder-Mead in linear coords
    if best_params_linear is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params_linear, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-19, 'fatol': 1e-21, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: One more L-BFGS-B with tightest tolerances
    if best_params_linear is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params_linear, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_linear = res.x.copy()
        except Exception:
            pass
    
    if best_params_linear is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params_linear)
# EVOLVE-BLOCK-END
