"""Rewritten program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_HAM_CACHE = {}

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
    key = (n, coupling, field)
    if key not in _HAM_CACHE:
        H = _build_hamiltonian(n, coupling, field)
        eigenvalues = np.linalg.eigvalsh(H)
        _HAM_CACHE[key] = (H, eigenvalues[0])
    return _HAM_CACHE[key][1]

def _circuit_state_fast(theta, n, layers):
    """
    Efficiently computes the state vector of the circuit.
    theta: (layers, n) array
    n: number of qubits
    layers: number of layers
    Returns: flat array of size 2**n
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            # Apply Ry on qubit i
            # Iterate over basis states where bit i is 0
            # For each such state b, the partner is b + (1<<i)
            for b in range(dim):
                if (b >> i) & 1 == 0:
                    b1 = b + (1 << i)
                    v0 = state[b]
                    v1 = state[b1]
                    state[b] = c * v0 - s * v1
                    state[b1] = s * v0 + c * v1
        
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                # Apply CZ: flip sign if both bits are 1
                for b in range(dim):
                    if (b & mask_i) and (b & mask_ip1):
                        state[b] *= -1.0
                        
    return state

def _energy_func(theta_flat, n, coupling, field, H, E0):
    """
    Energy evaluation for optimizer.
    theta_flat: 1D array of size layers*n
    """
    theta = theta_flat.reshape((3, n))
    state = _circuit_state_fast(theta, n, 3)
    # Energy = <psi|H|psi>
    # Since state is real, this is dot(state, H @ state)
    e_val = np.dot(state, H @ state)
    return e_val

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    H, E0 = _get_ground_energy(n, coupling, field)
    
    # Initial guess: product |+> state (seed)
    # This corresponds to theta[-1] = pi/2, others 0
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    
    best_theta = theta0.copy()
    best_energy = _energy_func(best_theta.flatten(), n, coupling, field, H, E0)
    
    # Strategy 1: Differential Evolution (Global Search)
    # DE is good for non-convex landscapes and doesn't need gradients.
    # Dimension is 3*n. For n=6, dim=18.
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    if time.time() < deadline - 30:
        try:
            # popsize=15, maxiter=50 should be fast enough for dim=18
            # Use seed for reproducibility
            de_result = opt.differential_evolution(
                _energy_func, 
                bounds, 
                args=(n, coupling, field, H, E0),
                maxiter=50, 
                popsize=15, 
                seed=42,
                tol=1e-6,
                init='sobol'
            )
            if de_result.fun < best_energy:
                best_energy = de_result.fun
                best_theta = de_result.x.reshape((layers, n))
        except Exception:
            pass

    # Strategy 2: Local Refinement with L-BFGS-B
    # Start from the best point found by DE
    if time.time() < deadline - 10:
        try:
            lbfgs_result = opt.minimize(
                _energy_func, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if lbfgs_result.fun < best_energy:
                best_energy = lbfgs_result.fun
                best_theta = lbfgs_result.x.reshape((layers, n))
        except Exception:
            pass

    # Strategy 3: Basin Hopping around the best point
    # Basin Hopping is effective for finding global minima in rugged landscapes.
    if time.time() < deadline - 5:
        try:
            # Define a step function for basin hopping
            def step(x):
                # Random perturbation
                return x + np.random.normal(0, 0.1, size=x.shape)
            
            bh_result = opt.minimize(
                _energy_func, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='Basinhopping',
                options={
                    'niter': 50, 
                    'T': 1.0, 
                    'stepsize': 0.5,
                    'minimize_kwargs': {'method': 'L-BFGS-B', 'options': {'maxiter': 100}}
                }
            )
            if bh_result.fun < best_energy:
                best_energy = bh_result.fun
                best_theta = bh_result.x.reshape((layers, n))
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 1:
        try:
            final_result = opt.minimize(
                _energy_func, 
                best_theta.flatten(), 
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if final_result.fun < best_energy:
                best_energy = final_result.fun
                best_theta = final_result.x.reshape((layers, n))
        except Exception:
            pass

    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
