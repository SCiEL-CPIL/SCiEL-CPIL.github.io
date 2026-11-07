"""Combined: root-space parameterization with feasibility projection and multi-start refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, brentq

# Constants from Hermite polynomial evaluations at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 2032128.0

def _Q_from_params(c0, c1, c2):
    """Return Q coefficients [a2, a4, a6, a8, a10, a12] from Hermite params."""
    c3 = -(c0 + H4_0*c1 + H8_0*c2) / H12_0
    a2  = -48.0*c1 - 13440.0*c2 - 1290240.0*c3
    a4  =  16.0*c1 + 11520.0*c2 + 2123328.0*c3
    a6  = -3072.0*c2 - 1548288.0*c3
    a8  =  256.0*c2 + 516096.0*c3
    a10 = -73728.0*c3
    a12 =  4096.0*c3
    return np.array([a2, a4, a6, a8, a10, a12])

def _value_from_Q(Q):
    """Compute value = r^2/(2*pi) where r^2 is largest positive sign-changing root of Q."""
    Q = Q.copy()
    if Q[5] < 0:
        Q = -Q
    coeffs = np.array([Q[5], Q[4], Q[3], Q[2], Q[1], Q[0]])
    # Trim leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    if len(coeffs) <= 1:
        return 1e10
    try:
        roots = np.roots(coeffs)
    except:
        return 1e10
    best_z = 0.0
    for rz in roots:
        if abs(rz.imag) > 1e-9:
            continue
        zr = rz.real
        if zr <= 1e-12:
            continue
        # Check sign change
        eps = 1e-7 * max(1.0, abs(zr))
        qm = np.polyval(coeffs, zr - eps)
        qp = np.polyval(coeffs, zr + eps)
        if qm * qp < 0 and zr > best_z:
            best_z = zr
    if best_z == 0.0:
        pos_roots = [rz.real for rz in roots if abs(rz.imag) < 1e-9 and rz.real > 1e-12]
        if pos_roots:
            best_z = max(pos_roots)
        else:
            return 1e10
    return best_z / (2.0 * np.pi)

def _objective(params):
    try:
        Q = _Q_from_params(params[0], params[1], params[2])
        v = _value_from_Q(Q)
        return v if np.isfinite(v) else 1e10
    except:
        return 1e10

def _feasibility_projection(c0, c1, c2, target_z):
    """
    Structural change: project (c0,c1,c2) so that Q(target_z) = 0.
    This constrains the search to the 2D manifold where target_z is a root,
    then we minimize the value directly on this manifold.
    """
    # Q(z) is linear in (c0,c1,c2,c3), and c3 is linear in (c0,c1,c2)
    # So Q(z) is linear in (c0,c1,c2)
    # We want to find the closest point on the hyperplane Q(target_z)=0
    # to the current (c0,c1,c2)
    
    # Compute gradient of Q(target_z) w.r.t. (c0,c1,c2)
    c3 = -(c0 + H4_0*c1 + H8_0*c2) / H12_0
    Q = _Q_from_params(c0, c1, c2)
    
    # Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    # Each ai is linear in (c0,c1,c2)
    # dQ/dci = sum_k dk_ai/dci * z^k
    
    # From _Q_from_params:
    # a2  = -48*c1 - 13440*c2 - 1290240*c3
    # a4  =  16*c1 + 11520*c2 + 2123328*c3
    # a6  = -3072*c2 - 1548288*c3
    # a8  =  256*c2 + 516096*c3
    # a10 = -73728*c3
    # a12 =  4096*c3
    # c3 = -(c0 + 12*c1 + 1680*c2)/2032128
    
    dc3_dc0 = -1.0/H12_0
    dc3_dc1 = -H4_0/H12_0
    dc3_dc2 = -H8_0/H12_0
    
    z = target_z
    z2 = z*z
    z3 = z2*z
    z4 = z2*z2
    z5 = z4*z
    
    # dQ/dci for i=0,1,2
    # a2: d/dci = [-1290240*dc3_dc0, -48-1290240*dc3_dc1, -13440-1290240*dc3_dc2]
    # a4: d/dci = [2123328*dc3_dc0, 16+2123328*dc3_dc1, 11520+2123328*dc3_dc2]
    # a6: d/dci = [-1548288*dc3_dc0, -1548288*dc3_dc1, -3072-1548288*dc3_dc2]
    # a8: d/dci = [516096*dc3_dc0, 516096*dc3_dc1, 256+516096*dc3_dc2]
    # a10: d/dci = [-73728*dc3_dc0, -73728*dc3_dc1, -73728*dc3_dc2]
    # a12: d/dci = [4096*dc3_dc0, 4096*dc3_dc1, 4096*dc3_dc2]
    
    grad = np.zeros(3)
    for k, zk in enumerate([1.0, z, z2, z3, z4, z5]):
        ai_grad = np.array([
            [-1290240*dc3_dc0, -48-1290240*dc3_dc1, -13440-1290240*dc3_dc2],
            [2123328*dc3_dc0, 16+2123328*dc3_dc1, 11520+2123328*dc3_dc2],
            [-1548288*dc3_dc0, -1548288*dc3_dc1, -3072-1548288*dc3_dc2],
            [516096*dc3_dc0, 516096*dc3_dc1, 256+516096*dc3_dc2],
            [-73728*dc3_dc0, -73728*dc3_dc1, -73728*dc3_dc2],
            [4096*dc3_dc0, 4096*dc3_dc1, 4096*dc3_dc2]
        ][k])
        grad += zk * ai_grad
    
    Q_val = np.polyval([Q[5], Q[4], Q[3], Q[2], Q[1], Q[0]], z)
    
    if abs(np.dot(grad, grad)) < 1e-30:
        return np.array([c0, c1, c2])
    
    # Project: new_params = params - Q_val/|grad|^2 * grad
    factor = Q_val / np.dot(grad, grad)
    return np.array([c0, c1, c2]) - factor * grad

def construct():
    t0 = time.time()
    deadline = t0 + 185.0  # 25s safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective(seed)
    best_params = seed.copy()
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 80.0:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=150, tol=1e-10,
                popsize=25, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, updating='immediate',
                workers=1
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement with feasibility projection
    np.random.seed(999)
    starts = [best_params.copy()]
    for scale in [0.001, 0.01, 0.05, 0.1, 0.5]:
        for _ in range(4):
            starts.append(best_params + np.random.randn(3) * scale)
    
    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12,
                                   'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    # Phase 3: Feasibility-projected refinement
    # For a range of target z values, project and refine
    if time.time() < deadline - 20.0:
        current_z = best_val * 2.0 * np.pi  # r^2 = z
        for target_z in [current_z * 0.5, current_z * 0.8, current_z * 1.2, current_z * 1.5]:
            if time.time() > deadline - 15.0:
                break
            projected = _feasibility_projection(best_params[0], best_params[1], best_params[2], target_z)
            try:
                res = minimize(_objective, projected, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-13,
                                       'fatol': 1e-15, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except:
                pass
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 10.0:
        for _ in range(2):
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 15000, 'xatol': 1e-14,
                                       'fatol': 1e-16, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except:
                break
    
    # Phase 5: Final L-BFGS attempt
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
