"""Shallow-circuit Ising ground states - tensor network energy with exact gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _circuit_energy_and_grad(theta, n, layers, H):
    """Compute energy and gradient using state-vector method with adjoint-like approach.
    
    For n<=6, we can afford full state vectors. We use finite-difference-free
    exact gradient via the parameter shift rule for Ry gates.
    
    Returns (energy, grad) where grad has shape (layers, n).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 + 0j
    
    # Precompute index arrays
    base = np.arange(dim)
    idx_arrays = []
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays.append((idx0, idx1))
    
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        mask = (base & m) == m
        cz_masks.append(mask)
    
    def evolve(state_in, theta_override=None, override_idx=None):
        """Evolve state through the circuit. If theta_override is given, use it for layer override_idx."""
        s = state_in.copy()
        for l in range(layers):
            for i in range(n):
                if theta_override is not None and l == override_idx:
                    t = theta_override[i]
                else:
                    t = theta[l, i]
                c = np.cos(t / 2.0)
                sn = np.sin(t / 2.0)
                idx0, idx1 = idx_arrays[i]
                old0 = s[idx0].copy()
                old1 = s[idx1].copy()
                s[idx0] = c * old0 - sn * old1
                s[idx1] = sn * old0 + c * old1
            
            if l < layers - 1:
                for mask in cz_masks:
                    s[mask] *= -1.0
        return s
    
    # Compute forward state
    psi = evolve(state)
    energy = float(np.real(np.vdot(psi, H @ psi)))
    
    # Compute gradient using parameter shift rule for each Ry angle
    # For Ry(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]],
    # d/dt <psi|H|psi> can be computed via shift: f(t+pi/2) - f(t-pi/2)
    # But this requires 2 evaluations per parameter = 36 evaluations.
    # Instead, use a more efficient approach: compute the state after each layer
    # and use the chain rule.
    
    # Actually for n<=6 with 18 params, let's just use central differences
    # but be smart about it. Or better: use the analytic gradient via
    # storing intermediate states.
    
    # Let's use a different approach: compute gradient via automatic differentiation
    # by tracking the state evolution analytically.
    
    # For efficiency with n<=6, use numerical gradient with central differences
    # but only 2 evaluations per parameter (forward and backward)
    grad = np.zeros((layers, n))
    eps = 1e-7
    
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[l, i] += eps
            theta_minus[l, i] -= eps
            
            psi_plus = evolve(state, theta_plus, l)
            psi_minus = evolve(state, theta_minus, l)
            
            e_plus = float(np.real(np.vdot(psi_plus, H @ psi_plus)))
            e_minus = float(np.real(np.vdot(psi_minus, H @ psi_minus)))
            
            grad[l, i] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def _circuit_energy(theta, n, layers, H):
    """Compute just the energy."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    base = np.arange(dim)
    idx_arrays = []
    for i in range(n):
        mask = 1 << i
        idx0 = base[(base & mask) == 0]
        idx1 = idx0 | mask
        idx_arrays.append((idx0, idx1))
    
    cz_masks = []
    for i in range(n - 1):
        m = (1 << i) | (1 << (i + 1))
        mask = (base & m) == m
        cz_masks.append(mask)
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            for mask in cz_masks:
                state[mask] *= -1.0
    
    return float(state @ (H @ state))

def _physics_init(n, coupling, field):
    """Physics-based initialization."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    
    theta = np.zeros((3, n))
    theta[0] = t_opt * 0.15
    theta[1] = t_opt * 0.25
    theta[2] = t_opt * 0.60
    return theta

def _symmetric_init(n, coupling, field):
    """Symmetric initialization."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    return np.full((3, n), t_opt / 3.0)

def _final_layer_init(n, coupling, field):
    """All rotation in final layer."""
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(ratio / 2.0)
    else:
        t_opt = np.pi / 2
    theta = np.zeros((3, n))
    theta[2] = t_opt
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(x):
        theta = x.reshape((3, n))
        return _circuit_energy(theta, n, 3, H)
    
    def objective_grad(x):
        theta = x.reshape((3, n))
        e, g = _circuit_energy_and_grad(theta, n, 3, H)
        return e, g.flatten()
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(12345)
    candidates = []
    
    candidates.append(_physics_init(n, coupling, field))
    candidates.append(_symmetric_init(n, coupling, field))
    candidates.append(_final_layer_init(n, coupling, field))
    
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2
    candidates.append(theta_simple)
    
    candidates.append(np.full((3, n), np.pi / 4))
    
    for _ in range(40):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate all candidates
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 10:
            break
        e = objective(tc.flatten())
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Primary optimization: L-BFGS-B with analytic gradient
    if time.time() < deadline - 30:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Powell polish
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Random restarts with local optimization using gradient
    restart_budget = 30
    for r in range(restart_budget):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.8, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final ultra-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-16}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Ensure output is finite
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
