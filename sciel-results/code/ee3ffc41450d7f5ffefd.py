"""Shallow-circuit Ising ground states - direct analytical approach with analytic gradients."""
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

def _precompute_masks(n):
    """Precompute bit masks and index arrays for fast state evolution."""
    dim = 1 << n
    masks = [1 << i for i in range(n)]
    idx0_cache = {}
    idx1_cache = {}
    for i in range(n):
        m = masks[i]
        idx0 = np.array([b for b in range(dim) if (b & m) == 0], dtype=int)
        idx1 = idx0 | m
        idx0_cache[i] = idx0
        idx1_cache[i] = idx1
    cz_indices = []
    for i in range(n - 1):
        mi = masks[i]
        mip1 = masks[i + 1]
        both_set = np.array([b for b in range(dim) if (b & mi) and (b & mip1)], dtype=int)
        cz_indices.append(both_set)
    return masks, idx0_cache, idx1_cache, cz_indices

def _circuit_state(theta, n, layers, precomp=None):
    """Compute the statevector of the 3-layer Ry/CZ circuit using vectorized ops."""
    dim = 1 << n
    if precomp is None:
        precomp = _precompute_masks(n)
    masks, idx0_cache, idx1_cache, cz_indices = precomp
    
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta[l,i]) to each qubit i
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0 = idx0_cache[i]
            idx1 = idx1_cache[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for both_set in cz_indices:
                state[both_set] *= -1.0
    
    return state

def _circuit_state_and_energy(theta, n, H, precomp=None):
    """Compute state and energy in one call."""
    state = _circuit_state(theta, n, 3, precomp)
    energy = float(np.dot(state, H @ state))
    return state, energy

def _circuit_energy_fast(theta, n, H, precomp=None):
    """Fast energy computation for a given set of angles."""
    _, energy = _circuit_state_and_energy(theta, n, H, precomp)
    return energy

def _energy_gradient(theta, n, H, precomp=None):
    """
    Compute gradient of energy w.r.t. angles using parameter-shift rule.
    For Ry(theta) = cos(theta/2)|0><0| - sin(theta/2)|1><0| - sin(theta/2)|0><1| + cos(theta/2)|1><1|
    d/dtheta <psi|H|psi> = -1/2 * <psi| [H, dRy/dtheta] |psi>
    Using parameter-shift: grad_i = (E(theta_i + pi/2) - E(theta_i - pi/2)) / 2
    """
    if precomp is None:
        precomp = _precompute_masks(n)
    grad = np.zeros_like(theta)
    for l in range(3):
        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[l, i] += np.pi / 2
            theta_minus[l, i] -= np.pi / 2
            e_plus = _circuit_energy_fast(theta_plus, n, H, precomp)
            e_minus = _circuit_energy_fast(theta_minus, n, H, precomp)
            grad[l, i] = (e_plus - e_minus) / 2.0
    return grad

def _energy_and_grad(theta_flat, n, H, precomp=None):
    """Return (energy, gradient) for scipy optimizers."""
    theta = theta_flat.reshape((3, n))
    energy = _circuit_energy_fast(theta, n, H, precomp)
    grad = _energy_gradient(theta, n, H, precomp)
    return energy, grad.flatten()

def _exact_ground_state(n, coupling, field):
    """Get exact ground state and energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return psi0, E0

def _mean_field_init(n, coupling, field):
    """
    Mean-field / variational product state initialization.
    For H = -J sum Z_i Z_{i+1} - h sum X_i, a product state |theta>^n
    has energy per spin: <E>/n = -J cos(theta)^2 - h sin(theta)
    """
    theta = np.zeros((3, n))
    
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(min(ratio / 2.0, 1.0))
    else:
        t_opt = np.pi / 2
    
    return theta, t_opt

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    psi0, E0 = _exact_ground_state(n, coupling, field)
    precomp = _precompute_masks(n)
    
    # Mean-field initialization
    theta_mf, t_opt = _mean_field_init(n, coupling, field)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Original mean-field distribution
    tc = theta_mf.copy()
    tc[0] = t_opt * 0.5
    tc[1] = t_opt * 0.3
    tc[2] = t_opt * 0.2
    candidates.append(tc)
    
    # Equal distribution
    tc = np.full((3, n), t_opt / 3.0)
    candidates.append(tc)
    
    # Front-loaded
    tc = np.zeros((3, n))
    tc[0] = t_opt * 0.7
    tc[1] = t_opt * 0.2
    tc[2] = t_opt * 0.1
    candidates.append(tc)
    
    # Back-loaded
    tc = np.zeros((3, n))
    tc[0] = t_opt * 0.1
    tc[1] = t_opt * 0.2
    tc[2] = t_opt * 0.7
    candidates.append(tc)
    
    # All in last layer
    tc = np.zeros((3, n))
    tc[-1] = np.pi / 2
    candidates.append(tc)
    
    # Uniform pi/4
    candidates.append(np.full((3, n), np.pi / 4))
    
    # Zero
    candidates.append(np.zeros((3, n)))
    
    # Small angles
    candidates.append(np.full((3, n), 0.2))
    
    # Additional candidates based on physics
    # For small field, angles should be small
    candidates.append(np.full((3, n), field / (2 * coupling + 1e-10)))
    # For large field, angles approach pi/2
    candidates.append(np.full((3, n), min(field / (coupling + 1e-10), np.pi/2)))
    
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline:
            break
        e = _circuit_energy_fast(tc, n, H, precomp)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Random starts with different seeds
    rng = np.random.default_rng(12345)
    for trial in range(25):
        if time.time() > deadline - 10:
            break
        tc = rng.uniform(-np.pi, np.pi, size=(3, n))
        e = _circuit_energy_fast(tc, n, H, precomp)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Perturbations around best
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        e = _circuit_energy_fast(tc, n, H, precomp)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Define objective and gradient functions for scipy
    def objective(x):
        theta = x.reshape((3, n))
        return _circuit_energy_fast(theta, n, H, precomp)
    
    def objective_grad(x):
        theta = x.reshape((3, n))
        return _energy_gradient(theta, n, H, precomp).flatten()
    
    def objective_and_grad(x):
        theta = x.reshape((3, n))
        e = _circuit_energy_fast(theta, n, H, precomp)
        g = _energy_gradient(theta, n, H, precomp)
        return e, g.flatten()
    
    # Strategy 1: L-BFGS-B with analytic gradients - main optimization
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(
                objective_and_grad,
                best_theta.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 2: Basin Hopping for global exploration
    if time.time() < deadline - 15:
        try:
            def bh_objective(x):
                theta = x.reshape((3, n))
                return _circuit_energy_fast(theta, n, H, precomp)
            
            res = opt.basinhopping(
                bh_objective,
                best_theta.flatten(),
                niter=30,
                T=1.0,
                stepsize=0.5,
                minimizer_kwargs={'method': 'L-BFGS-B', 'jac': objective_grad},
                acceptance='all',
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 3: Try optimization from second-best candidate
    second_best_theta = None
    second_best_energy = float('inf')
    for tc in candidates:
        if time.time() > deadline - 15:
            break
        e = _circuit_energy_fast(tc, n, H, precomp)
        if e < second_best_energy and e >= best_energy - 1e-10:
            if np.allclose(tc, best_theta, atol=1e-6):
                continue
            second_best_energy = e
            second_best_theta = tc.copy()
    
    if second_best_theta is not None and time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective_and_grad,
                second_best_theta.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 4: Nelder-Mead for polish (derivative-free)
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 5: Final L-BFGS-B polish with very tight tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective_and_grad,
                best_theta.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-20, 'gtol': 1e-18}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final: try a few more random restarts with optimization if time permits
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        try:
            res = opt.minimize(
                objective_and_grad,
                tc.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
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
