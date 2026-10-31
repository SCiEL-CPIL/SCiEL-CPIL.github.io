"""Analytic Hermite-Gaussian uncertainty bound with generalized eigenvalue + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

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

    # Gradient using quotient rule
    # d(m2/m0)/da_i = (dm2_da_i * m0 - m2 * dm0_da_i) / m0^2
    # d(bound)/da_i = 0.5/bound * (vF * d(m2/m0)/da_i + (m2/m0) * d(n2/n0)/da_i)
    # But vF = 7/3 is constant, so d(vF)/da_i = 0
    # Thus: grad = 0.5/bound * vF * d(m2/m0)/da_i
    
    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    # Since vF is constant, only the m2/m0 part contributes to the gradient
    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)

    grad_da0 = 0.5 / bound * vF * dvf_da0
    grad_da4 = 0.5 / bound * vF * dvf_da4
    grad_da8 = 0.5 / bound * vF * dvf_da8

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _objective_unit(params):
    """Objective on unit sphere: normalize params and compute bound."""
    norm = np.linalg.norm(params)
    if norm < 1e-15:
        return 1e18, np.array([0.0, 0.0, 0.0])
    unit = params / norm
    g2 = 2.0 * unit[0] + 210.0 * unit[2]
    if abs(g2) < 1e-12:
        return 1e18, np.array([0.0, 0.0, 0.0])
    val, grad = _bound_and_grad(unit[0], unit[1], unit[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    # Project gradient onto tangent space of unit sphere
    grad_tangent = grad - np.dot(grad, unit) * unit
    return val, grad_tangent


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    """
    Key insight: vF = n2/n0 = 7/3 is constant (given g(0)=0 constraint).
    So bound = sqrt(7/3 * m2/m0). Minimizing m2/m0 is a Rayleigh quotient problem.
    Set up symmetric matrices A, B such that m0_inner = x^T A x, m2_inner = x^T B x.
    Solve B v = lambda A v; smallest eigenvalue gives the minimum ratio.
    
    The problem is scale-invariant, so we can fix the scale and optimize over the 2D sphere.
    """
    # Build symmetric matrices A and B
    # m0_inner = a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2
    # where a12 = -(a0+3*a4+105*a8)/945
    # 945*a12^2 = (a0+3*a4+105*a8)^2 / 945
    # Expanding: (a0^2 + 9*a4^2 + 11025*a8^2 + 6*a0*a4 + 210*a0*a8 + 630*a4*a8)/945

    A = np.array([
        [1.0 + 1.0/945.0,       3.0/945.0,       105.0/945.0],
        [3.0/945.0,              3.0 + 9.0/945.0, 315.0/945.0],
        [105.0/945.0,            315.0/945.0,     105.0 + 11025.0/945.0]
    ])

    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1680*a4*a8 + 2940*a8^2 + 11340*a8*a12
    # 11340*a8*a12 = -12*a8*a0 - 36*a8*a4 - 1260*a8^2
    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1644*a4*a8 + 1680*a8^2 - 12*a0*a8

    B = np.array([
        [3.0,     12.0,   -6.0],
        [12.0,    105.0,  822.0],
        [-6.0,    822.0,  1680.0]
    ])

    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])

    # Solve generalized eigenvalue problem B v = lambda A v
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    # Get all eigenvalue solutions
    eigenvalue_solutions = []
    try:
        eigenvalues, eigenvectors = eigh(B, A)
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            g2_v = 2.0 * v[0] + 210.0 * v[2]
            if abs(g2_v) > 1e-12:
                # Scale to unit norm
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-15:
                    v_scaled = v / norm_v
                    eigenvalue_solutions.append(v_scaled)
                    val_check, _ = _bound_and_grad(*v_scaled)
                    if np.isfinite(val_check) and val_check < best_val:
                        best_val = val_check
                        best = v_scaled.copy()
    except Exception:
        pass

    # Also try the seed scaled to unit norm
    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_unit = seed / norm_seed
        val_seed_unit, _ = _bound_and_grad(*seed_unit)
        if np.isfinite(val_seed_unit) and val_seed_unit < best_val:
            best_val = val_seed_unit
            best = seed_unit.copy()

    # Multi-start L-BFGS-B on the unit sphere (scale-invariant problem)
    rng = np.random.default_rng(42)
    
    # Generate multi-start points on the unit sphere in 3D
    sphere_starts = []
    
    # Normalize best to unit sphere
    norm_best = np.linalg.norm(best)
    if norm_best > 1e-15:
        sphere_starts.append(best / norm_best)
    
    # Add eigenvalue solutions (already unit norm)
    for v in eigenvalue_solutions:
        sphere_starts.append(v.copy())
    
    # Add random points on the unit sphere
    for i in range(80):
        rand_dir = rng.normal(0.0, 1.0, 3)
        norm_rd = np.linalg.norm(rand_dir)
        if norm_rd > 1e-15:
            rand_unit = rand_dir / norm_rd
            g2_rand = 2.0 * rand_unit[0] + 210.0 * rand_unit[2]
            if abs(g2_rand) > 1e-12:
                sphere_starts.append(rand_unit)
    
    # Add structured perturbations around the best solution
    for i in range(40):
        scale = 0.5 / (i + 1)
        perturbed = best / norm_best + rng.normal(0.0, scale, 3)
        norm_p = np.linalg.norm(perturbed)
        if norm_p > 1e-15:
            perturbed_unit = perturbed / norm_p
            g2_p = 2.0 * perturbed_unit[0] + 210.0 * perturbed_unit[2]
            if abs(g2_p) > 1e-12:
                sphere_starts.append(perturbed_unit)

    # Optimize on the unit sphere using projected gradient
    for s in sphere_starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_unit, s, method='L-BFGS-B',
                           options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                # Normalize result back to unit sphere
                norm_res = np.linalg.norm(res.x)
                if norm_res > 1e-15:
                    best = res.x / norm_res
                    best_val = res.fun
        except Exception:
            pass

    # Newton polish on the unit sphere
    if time.time() < _DEADLINE - 15:
        try:
            # Compute Hessian via finite differences of the projected gradient
            def _newton_step(x):
                f, g = _objective_unit(x)
                if not np.isfinite(f):
                    return x, f
                # Central difference Hessian
                h = 1e-5
                H = np.zeros((3, 3))
                for i in range(3):
                    ei = np.zeros(3)
                    ei[i] = h
                    f_plus, g_plus = _objective_unit(x + ei)
                    f_minus, g_minus = _objective_unit(x - ei)
                    for j in range(3):
                        if np.isfinite(f_plus) and np.isfinite(f_minus):
                            H[i, j] = (g_plus[j] - g_minus[j]) / (2.0 * h)
                # Solve H dx = -g for Newton step
                try:
                    dx = np.linalg.solve(H + 1e-10 * np.eye(3), -g)
                except np.linalg.LinAlgError:
                    dx = -g / (np.linalg.norm(g) + 1e-15)
                # Project dx onto tangent space
                dx_tangent = dx - np.dot(dx, x) * x
                # Line search
                step = 1.0
                best_x = x
                best_f = f
                for _ in range(50):
                    trial = x + step * dx_tangent
                    norm_trial = np.linalg.norm(trial)
                    if norm_trial > 1e-15:
                        trial = trial / norm_trial
                    f_trial, _ = _objective_unit(trial)
                    if np.isfinite(f_trial) and f_trial < best_f:
                        best_x = trial
                        best_f = f_trial
                        break
                    step *= 0.5
                return best_x, best_f
            
            # Multiple Newton iterations
            for _ in range(20):
                if time.time() > _DEADLINE - 10:
                    break
                new_x, new_f = _newton_step(best)
                if np.isfinite(new_f) and new_f < best_val:
                    best = new_x
                    best_val = new_f
                else:
                    break
        except Exception:
            pass

    # Final verification and normalization
    if time.time() < _DEADLINE - 5:
        norm_final = np.linalg.norm(best)
        if norm_final > 1e-15:
            best = best / norm_final
        val_final, _ = _bound_and_grad(*best)
        if np.isfinite(val_final) and val_final < best_val:
            best_val = val_final

    return np.array(best)
# EVOLVE-BLOCK-END
