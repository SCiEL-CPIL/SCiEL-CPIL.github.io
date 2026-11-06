"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_(i+1) - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_(i+1) terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            H[idx, flipped] += -field
    
    return H


def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _apply_ry_vectorized(state, theta, qubit, n):
    """Apply Ry(theta) to qubit using vectorized operations."""
    dim = 2 ** n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    mask = 1 << qubit
    idx = np.arange(dim)
    
    is_0 = (idx & mask) == 0
    is_1 = (idx & mask) == mask
    
    result = np.zeros_like(state)
    
    idx0_0 = idx[is_0]
    idx1_0 = idx0_0 | mask
    v0 = state[idx0_0]
    v1 = state[idx1_0]
    result[idx0_0] = c * v0 - s * v1
    result[idx1_0] = s * v0 + c * v1
    
    return result


def _apply_cz_vectorized(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2 using vectorized operations."""
    dim = 2 ** n
    mask1 = 1 << q1
    mask2 = 1 << q2
    
    idx = np.arange(dim)
    both_1 = ((idx & mask1) == mask1) & ((idx & mask2) == mask2)
    
    result = state.copy()
    result[both_1] *= -1.0
    
    return result


def _build_circuit_state_fast(theta, n, layers):
    """Build the state vector from the circuit parameters using vectorized gates."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            state = _apply_ry_vectorized(state, theta[layer, q], q, n)
        # Apply CZ between this layer and next (if not last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_vectorized(state, q, q + 1, n)
    
    return state


def _compute_energy_fast(state, H):
    """Compute expectation value using pre-built Hamiltonian."""
    return np.real(np.vdot(state, H @ state))


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Get ground energy and Hamiltonian
    ground_energy = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state_fast(theta, n, layers)
        energy = _compute_energy_fast(state, H)
        return (energy - ground_energy) / n
    
    best_val = np.inf
    best_result = None
    
    # Exploit reflection symmetry: the Ising chain is symmetric under qubit reversal.
    # So theta[l, q] should equal theta[l, n-1-q].
    # This reduces parameters from layers*n to layers*ceil(n/2).
    # For n=4: layers*2 = 6 params. For n=6: layers*3 = 9 params.
    #
    # Additionally, for open chains with uniform coupling and field,
    # the optimal angles may have additional structure.
    # Let's use the reflection symmetry and do a grid search + local polish.
    
    # Determine the number of unique angles per layer based on reflection symmetry
    n_unique = (n + 1) // 2  # ceil(n/2)
    n_params = layers * n_unique
    
    def expand_params(params):
        """Expand symmetric parameters to full (layers, n) array."""
        theta = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n_unique):
                val = params[l * n_unique + q]
                theta[l, q] = val
                theta[l, n - 1 - q] = val
        return theta
    
    def objective_sym(params):
        theta = expand_params(params)
        state = _build_circuit_state_fast(theta, n, layers)
        energy = _compute_energy_fast(state, H)
        return (energy - ground_energy) / n
    
    # Grid search over the reduced parameter space
    # Use a coarse grid first, then refine
    
    # For the grid, we'll sample each unique angle in [-pi, pi]
    # Start with a moderately fine grid
    
    n_grid = 12  # points per angle
    grid_vals = np.linspace(-np.pi, np.pi, n_grid)
    
    # Generate grid points
    grid_points = np.array(np.meshgrid(*([grid_vals] * n_params))).T.reshape(-1, n_params)
    
    # Evaluate all grid points
    best_grid_val = np.inf
    best_grid_point = None
    
    for i, pt in enumerate(grid_points):
        if time.time() > deadline:
            break
        try:
            val = objective_sym(pt)
            if val < best_grid_val:
                best_grid_val = val
                best_grid_point = pt.copy()
        except Exception:
            pass
    
    # Refine around the best grid point with a finer grid
    if best_grid_point is not None:
        # Create a finer grid around the best point
        step = 2 * np.pi / (n_grid - 1)
        fine_n_grid = 20
        fine_step = step / fine_n_grid
        
        # Define bounds for refinement
        lower = best_grid_point - step
        upper = best_grid_point + step
        lower = np.maximum(lower, -np.pi)
        upper = np.minimum(upper, np.pi)
        
        fine_grid_vals = [np.linspace(lower[i], upper[i], fine_n_grid) for i in range(n_params)]
        fine_grid_points = np.array(np.meshgrid(*fine_grid_vals)).T.reshape(-1, n_params)
        
        for i, pt in enumerate(fine_grid_points):
            if time.time() > deadline:
                break
            try:
                val = objective_sym(pt)
                if val < best_grid_val:
                    best_grid_val = val
                    best_grid_point = pt.copy()
            except Exception:
                pass
    
    # Local optimization from the best grid point
    if best_grid_point is not None:
        try:
            res = minimize(objective_sym, best_grid_point, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * n_params,
                          options={'maxiter': 3000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_point = res.x
        except Exception:
            pass
    
    # Also try a few other starting points for local optimization
    # Starting point: all zeros
    x0_zero = np.zeros(n_params)
    try:
        res = minimize(objective_sym, x0_zero, method='L-BFGS-B',
                      bounds=[(-np.pi, np.pi)] * n_params,
                      options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < best_grid_val:
            best_grid_val = res.fun
            best_grid_point = res.x
    except Exception:
        pass
    
    # Starting point: pi/2 in last layer (seed behavior)
    x0_pi2 = np.zeros(n_params)
    x0_pi2[(layers-1)*n_unique:] = np.pi / 2
    try:
        res = minimize(objective_sym, x0_pi2, method='L-BFGS-B',
                      bounds=[(-np.pi, np.pi)] * n_params,
                      options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < best_grid_val:
            best_grid_val = res.fun
            best_grid_point = res.x
    except Exception:
        pass
    
    # Starting point: pi/4 in all layers
    x0_pi4 = np.full(n_params, np.pi / 4)
    try:
        res = minimize(objective_sym, x0_pi4, method='L-BFGS-B',
                      bounds=[(-np.pi, np.pi)] * n_params,
                      options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < best_grid_val:
            best_grid_val = res.fun
            best_grid_point = res.x
    except Exception:
        pass
    
    # Starting point: -pi/4 in all layers
    x0_negpi4 = np.full(n_params, -np.pi / 4)
    try:
        res = minimize(objective_sym, x0_negpi4, method='L-BFGS-B',
                      bounds=[(-np.pi, np.pi)] * n_params,
                      options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < best_grid_val:
            best_grid_val = res.fun
            best_grid_point = res.x
    except Exception:
        pass
    
    # Add some random starting points
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        x0_rand = np.random.uniform(-np.pi, np.pi, size=n_params)
        try:
            res = minimize(objective_sym, x0_rand, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * n_params,
                          options={'maxiter': 1500, 'maxfun': 4000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_point = res.x
        except Exception:
            pass
    
    if best_grid_point is None:
        # Fallback: seed solution
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    # Expand the best symmetric parameters to full (layers, n) array
    theta = expand_params(best_grid_point)
    
    # Final local optimization on the full parameter space
    try:
        res = minimize(objective, theta.flatten(), method='L-BFGS-B',
                      bounds=[(-np.pi, np.pi)] * (layers * n),
                      options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-14})
        if res.fun < best_grid_val:
            return res.x.reshape((layers, n))
    except Exception:
        pass
    
    return theta
# EVOLVE-BLOCK-END
