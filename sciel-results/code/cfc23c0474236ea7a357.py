"""Combined program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
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

def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def _apply_cz_layer(state, n):
    dim = 1 << n
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for b in range(dim):
            if (b & mask_i) and (b & mask_ip1):
                state[b] *= -1.0
    return state

def _circuit_state(theta, n, layers):
    dim = 1 << n
    state = np.zeros((2,) * n, dtype=float)
    state[(0,) * n] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            state = np.moveaxis(state, i, 0)
            new_shape = (2,) + state.shape[1:]
            state = state.reshape(new_shape)
            
            old0 = state[0].copy()
            old1 = state[1].copy()
            state[0] = c * old0 - s * old1
            state[1] = s * old0 + c * old1
            
            state = state.reshape((2,) * n)
            state = np.moveaxis(state, 0, i)
        
        if l < layers - 1:
            state_flat = state.flatten()
            state_flat = _apply_cz_layer(state_flat, n)
            state = state_flat.reshape((2,) * n)
            
    return state.flatten()

def _energy(theta_flat, n, coupling, field, H, E0):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    e_val = np.dot(state, H @ state)
    return e_val

def _energy_and_grad(theta_flat, n, coupling, field):
    """Compute energy and gradient using adjoint method (analytical gradient)."""
    theta = theta_flat.reshape((3, n))
    layers = 3
    dim = 1 << n
    
    H = _build_hamiltonian(n, coupling, field)
    
    # Forward pass: compute state and store intermediate states
    # State is represented as a flat vector of length dim
    def _forward_and_store(theta):
        """Forward pass storing states after each Ry gate."""
        state = np.zeros(dim, dtype=float)
        state[0] = 1.0  # |0>^n
        stored = []  # list of (layer, qubit, state_before_ry)
        
        for l in range(layers):
            for i in range(n):
                stored.append((l, i, state.copy()))
                t = theta[l, i]
                c = np.cos(t / 2.0)
                s = np.sin(t / 2.0)
                state = _apply_ry(state, i, c, s, n)
            if l < layers - 1:
                state = _apply_cz_layer(state, n)
        return state, stored
    
    def _apply_ry(state, qubit, c, s, n):
        """Apply Ry(c,s) to qubit in flat state vector."""
        dim = 1 << n
        new_state = np.empty_like(state)
        mask = 1 << qubit
        for b in range(dim):
            if b & mask:
                # b has qubit=1, partner has qubit=0
                partner = b ^ mask
                new_state[b] = s * state[partner] + c * state[b]
                new_state[partner] = -s * state[b] + c * state[partner]
            # else: b has qubit=0, handled when we hit partner
        return new_state
    
    state_final, stored = _forward_and_store(theta)
    energy = np.dot(state_final, H @ state_final)
    
    # Adjoint method for gradient
    # dE/dtheta[l,i] = 2 * Re(<psi_final| H | dpsi/dtheta[l,i]>)
    # Using adjoint: propagate adjoint state backward
    # adjoint_final = H @ psi_final (since dE = 2 Re(<psi|H|dpsi>))
    adjoint = H @ state_final  # shape (dim,)
    
    grad = np.zeros_like(theta_flat)
    
    # Backward pass
    for idx in range(len(stored) - 1, -1, -1):
        l, i, state_before = stored[idx]
        
        # dpsi/dtheta for Ry at (l,i):
        # Ry(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        # dRy/dt = 1/2 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
        t = theta[l, i]
        c = np.cos(t / 2.0)
        s = np.sin(t / 2.0)
        
        # dpsi = (1/2) * Ry_d @ psi_before where Ry_d = [[-s, -c], [c, -s]]
        # So dpsi[qubit=0] = 1/2 * (-s * psi_before[0] - c * psi_before[1])
        #    dpsi[qubit=1] = 1/2 * ( c * psi_before[0] - s * psi_before[1])
        mask = 1 << i
        dpsi = np.zeros(dim, dtype=float)
        for b in range(dim):
            if b & mask:
                partner = b ^ mask
                # b has qubit=1
                dpsi[b] = 0.5 * (c * state_before[partner] - s * state_before[b])
                # partner has qubit=0
                dpsi[partner] = 0.5 * (-s * state_before[partner] - c * state_before[b])
        
        # Gradient contribution: 2 * Re(<adjoint | dpsi>)
        grad_flat = idx  # theta_flat index for (l, i)
        flat_idx = l * n + i
        grad[flat_idx] = 2.0 * np.dot(adjoint, dpsi)
        
        # Propagate adjoint backward through Ry
        # adjoint_before = Ry(t)^T @ adjoint_after
        # But we need adjoint_after for this gate, which is the current adjoint
        # adjoint_before = [[c, s], [-s, c]] @ adjoint
        new_adjoint = np.empty_like(adjoint)
        for b in range(dim):
            if b & mask:
                partner = b ^ mask
                new_adjoint[partner] = c * adjoint[partner] + s * adjoint[b]
                new_adjoint[b] = -s * adjoint[partner] + c * adjoint[b]
            else:
                new_adjoint[b] = adjoint[b]  # will be overwritten when partner is processed
        # Actually need to be more careful - let me redo
        new_adjoint = np.zeros(dim, dtype=float)
        for b in range(dim):
            if b & mask:
                partner = b ^ mask
                # Ry^T = [[c, s], [-s, c]]
                # adjoint_before[0] = c * adjoint_after[0] + s * adjoint_after[1]
                # adjoint_before[1] = -s * adjoint_after[0] + c * adjoint_after[1]
                new_adjoint[partner] = c * adjoint[partner] + s * adjoint[b]
                new_adjoint[b] = -s * adjoint[partner] + c * adjoint[b]
        adjoint = new_adjoint
        
        # Propagate adjoint backward through CZ layer (if applicable)
        # CZ is self-adjoint and real, so CZ^T = CZ
        # Check if next operation backward is a CZ
        # After gate (l, i), if i == n-1 and l < layers-1, then CZ was applied
        if i == n - 1 and l < layers - 1:
            # CZ is its own inverse and self-adjoint
            adjoint = _apply_cz_layer(adjoint, n)
    
    return energy, grad

def _physics_init(n, field):
    """Physics-inspired initialization based on arctan(1/field)."""
    theta = np.zeros((3, n))
    base_angle = np.arctan(1.0 / field)
    theta[0] = base_angle * 0.3
    theta[1] = base_angle * 0.5
    theta[2] = base_angle
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _get_ground_energy(n, coupling, field)
    
    # Multiple initial guesses
    theta0_seed = np.zeros((layers, n))
    theta0_seed[-1] = np.pi / 2
    
    theta0_physics = _physics_init(n, field)
    
    best_theta = theta0_seed.copy()
    best_energy = _energy(theta0_seed.flatten(), n, coupling, field, H, E0)
    
    # Try physics init as well
    e_phys = _energy(theta0_physics.flatten(), n, coupling, field, H, E0)
    if e_phys < best_energy:
        best_energy = e_phys
        best_theta = theta0_physics.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: L-BFGS-B with analytical gradient from multiple starting points
    starts = [theta0_seed, theta0_physics]
    
    # Add structured random starts
    for r in range(10):
        if time.time() > deadline - 15:
            break
        
        if r % 3 == 0:
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
        elif r % 3 == 1:
            current_theta = best_theta + rng.normal(0, 0.3, size=(layers, n))
        else:
            # Small perturbation around physics init
            current_theta = theta0_physics + rng.normal(0, 0.2, size=(layers, n))
        starts.append(current_theta)
    
    def objective_with_grad(x):
        """Objective function with gradient for L-BFGS-B."""
        e, g = _energy_and_grad(x, n, coupling, field)
        return e, g
    
    for st in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = opt.minimize(
                objective_with_grad, 
                st.flatten(), 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
        
        # Additional L-BFGS-B polish from the result of the first L-BFGS-B
        if time.time() < deadline - 15:
            try:
                res2 = opt.minimize(
                    objective_with_grad, 
                    best_theta.flatten(), 
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res2.fun < best_energy:
                    best_energy = res2.fun
                    best_theta = res2.x.reshape((layers, n))
            except Exception:
                pass
    
    # Strategy 2: Nelder-Mead polish from best
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from best using gradient
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective_with_grad, 
                best_theta.flatten(), 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ultra-precision final polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _energy, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0_seed
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
