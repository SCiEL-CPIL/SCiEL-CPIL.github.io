import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_truncated_octahedron(r1, r2, r3, r4):
    """
    Generate 38 atoms in a truncated octahedron geometry.
    
    Orbits under Oh symmetry:
    - 6 atoms at (±r1, 0, 0), (0, ±r1, 0), (0, 0, ±r1)  [face centers]
    - 12 atoms at (±r2, ±r2, 0) and permutations           [edge centers]
    - 12 atoms at (±r3, ±r3, ±r3)                          [body center region]
    - 8 atoms at (±r4, 0, ±r4) and permutations            [vertex-like]
    
    Total: 6 + 12 + 12 + 8 = 38
    """
    pts = []
    
    # Orbit 1: 6 face-center atoms
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * r1
            pts.append(p)
    
    # Orbit 2: 12 edge-center atoms
    for axis1 in range(3):
        for axis2 in range(3):
            if axis1 == axis2:
                continue
            for sign1 in [-1, 1]:
                for sign2 in [-1, 1]:
                    p = np.zeros(3)
                    p[axis1] = sign1 * r2
                    p[axis2] = sign2 * r2
                    pts.append(p)
    
    # Orbit 3: 12 body-center region atoms
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                p = np.array([s1 * r3, s2 * r3, s3 * r3])
                pts.append(p)
    
    # Orbit 4: 8 vertex-like atoms
    for axis in range(3):
        for axis2 in range(3):
            if axis == axis2:
                continue
            for sign1 in [-1, 1]:
                for sign2 in [-1, 1]:
                    p = np.zeros(3)
                    p[axis] = sign1 * r4
                    p[axis2] = sign2 * r4
                    pts.append(p)
    
    pos = np.array(pts, dtype=np.float64)
    pos -= np.mean(pos, axis=0)
    return pos

def _optimize_truncated_octahedron():
    """
    Optimize the 4 radial parameters of the truncated octahedron.
    """
    # Initial guess: scale so that nearest neighbors are at 2^(1/6)
    # For a regular truncated octahedron, the optimal geometry has specific ratios.
    # Let's start with a reasonable guess and let the optimizer find the best.
    # The face-center to edge-center distance should be ~2^(1/6)
    # r1 (face center), r2 (edge center): distance = sqrt(2)*r2
    # So r2 = 2^(1/6) / sqrt(2) ≈ 0.7937
    # r1 should be such that face-center to edge-center distance is also reasonable
    # distance from (r1,0,0) to (r2,r2,0) = sqrt((r1-r2)^2 + r2^2)
    # Let's just start with all parameters around 0.8-1.2 and let optimizer work
    
    x0 = np.array([1.0, 0.8, 0.7, 0.9])
    
    def obj(x):
        try:
            pos = _make_truncated_octahedron(x[0], x[1], x[2], x[3])
            e, _ = _lj_energy_and_grad(pos)
            return e
        except Exception:
            return 1e10
    
    # Use Nelder-Mead for this small 4D problem
    try:
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        best_x = res.x
        best_e = res.fun
        
        # Fine-tune with L-BFGS if we have a good starting point
        def obj_grad(x):
            pos = _make_truncated_octahedron(x[0], x[1], x[2], x[3])
            e, _ = _lj_energy_and_grad(pos)
            # Numerical gradient for the 4 parameters
            eps = 1e-7
            grad = np.zeros(4)
            for i in range(4):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                e_plus = obj(x_plus)
                e_minus = obj(x_minus)
                grad[i] = (e_plus - e_minus) / (2 * eps)
            return e, grad
        
        res2 = minimize(obj_grad, best_x, jac=True, method='L-BFGS-B',
                        options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14})
        if res2.fun < best_e:
            best_x = res2.x
            best_e = res2.fun
        
        pos = _make_truncated_octahedron(best_x[0], best_x[1], best_x[2], best_x[3])
        return pos, best_e, best_x
        
    except Exception:
        # Fallback: use initial guess
        pos = _make_truncated_octahedron(*x0)
        e, _ = _lj_energy_and_grad(pos)
        return pos, e, x0

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, n_iter=50):
    """
    1D global search over uniform scale factor s applied to centered positions.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.9, 1.1, 41):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.1 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def _basin_hopping(pos0, max_iterations=None, deadline=None):
    """
    Basin hopping algorithm to escape local minima.
    """
    rng = np.random.default_rng(42)
    
    current_pos = pos0.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_size = 0.5
    T = 1.0
    iteration = 0
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break
        
        iteration += 1
        
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_pos, new_energy = _relax(perturbed, max_iter=300, gtol=1e-6, ftol=1e-10)
        
        # Check separation constraint
        new_pos = _ensure_min_sep(new_pos)
        new_energy, _ = _lj_energy_and_grad(new_pos)
        
        # Acceptance Criterion
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            if rng.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy = new_energy
        
        # Adaptive step size: reduce if we're finding improvements, increase if stuck
        # Simple heuristic: every 50 iterations, adjust step size
        if iteration % 50 == 0:
            if current_energy < best_energy + 0.1:
                step_size *= 0.95
            else:
                step_size *= 1.05
            step_size = np.clip(step_size, 0.1, 2.0)
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    
    # 1. Generate initial truncated octahedron geometry with optimized parameters
    try:
        pos0, e0, params0 = _optimize_truncated_octahedron()
        # Ensure minimum separation
        pos0 = _ensure_min_sep(pos0)
        e0, _ = _lj_energy_and_grad(pos0)
    except Exception:
        # Fallback: random cluster
        rng = np.random.default_rng(42)
        pos0 = rng.uniform(-2, 2, size=(N, 3))
        pos0 = _ensure_min_sep(pos0)
        e0, _ = _lj_energy_and_grad(pos0)

    best_pos = pos0
    best_energy = e0

    # 2. Initial high-precision relaxation
    try:
        p, e = _relax(pos0, max_iter=1000, gtol=1e-10, ftol=1e-14)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # 3. 1D scale optimization
    if time.time() - t0 < DEADLINE - 20.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # 4. Basin Hopping phase
    if time.time() - t0 < DEADLINE - 10.0:
        try:
            bh_pos, bh_energy = _basin_hopping(best_pos, deadline=DEADLINE - 5.0)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass

    # 5. Final High-Precision Polish
    if time.time() - t0 < DEADLINE - 2.0:
        try:
            # First, try to optimize scale of the best found
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            
            # Then, final relaxation
            p_final, e_final = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-16)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = p_final
        except Exception:
            pass

    # Ensure final separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
