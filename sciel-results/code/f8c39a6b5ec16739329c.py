"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Multi-start local optimization with diverse physical initializers.
Instead of relying on a single mean-field starting point, we generate multiple
initial guesses based on different physical regimes (strong coupling, strong field,
intermediate, random) and run a fast gradient-based optimizer from each.
We keep the best result found across all starts within the time budget.
This explores the multi-modal landscape more effectively than a single start.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    # We store the parts separately to allow efficient energy calculation for different fields.
    # H_IZ = sum Z_i Z_(i+1)
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    # H_IX = sum X_i
    H_IX = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    key = (n, coupling, field)
    if key in _EE_CACHE:
        return _EE_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _EE_CACHE[key] = e0
    return e0

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    state = _apply_circuit(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def _generate_initializers(n, coupling, field, layers):
    """
    Generate a diverse set of initial parameter arrays for multi-start optimization.
    """
    initializers = []
    
    # 1. Mean-field product state approximation
    def prod_energy(theta):
        return -coupling * (n - 1) * np.cos(theta)**2 - field * n * np.sin(theta)
    
    thetas = np.linspace(-np.pi, np.pi, 1000)
    energies = np.array([prod_energy(t) for t in thetas])
    idx_min = np.argmin(energies)
    theta_opt = thetas[idx_min]
    res = minimize(prod_energy, theta_opt, method='BFGS')
    if res.success:
        theta_opt = res.x[0]
    initializers.append(np.full((layers, n), theta_opt))
    
    # 2. Strong coupling limit: all spins aligned (theta = 0)
    initializers.append(np.zeros((layers, n)))
    
    # 3. Strong field limit: all spins in x-basis (theta = pi/2)
    initializers.append(np.full((layers, n), np.pi / 2))
    
    # 4. Alternating pattern for antiferromagnetic-like correlations
    alt = np.tile(np.array([0, np.pi]), n)[:n]
    initializers.append(np.tile(alt, (layers, 1)))
    
    # 5. Random initializations (a few of them)
    rng = np.random.default_rng(42)
    for _ in range(3):
        initializers.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
    
    return initializers

def _objective(params, n, coupling, field, e_exact):
    """Objective function for the optimizer."""
    theta = params.reshape(3, n)
    e_circ = _compute_energy(theta, n, coupling, field)
    return e_circ - e_exact

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Compute exact ground energy
    e_exact = _compute_exact_ground_energy(n, coupling, field)
    
    # 2. Generate diverse initializers
    initializers = _generate_initializers(n, coupling, field, layers)
    
    best_x = initializers[0].flatten()
    best_val = _objective(best_x, n, coupling, field, e_exact)
    
    # 3. Run optimization from each initial point
    for init in initializers:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        current_val = _objective(x0, n, coupling, field, e_exact)
        
        # Skip if this starting point is already much worse and time is tight
        # But we still try to optimize from it as it might lead to a different basin
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(n, coupling, field, e_exact),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
        except Exception:
            continue
            
        # Check time again after each start
        if time.time() > deadline - 5.0:
            break
            
    # 4. Final polish on the best result if time permits
    if time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                _objective, 
                best_x, 
                args=(n, coupling, field, e_exact),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    theta_opt = best_x.reshape(layers, n)
    return theta_opt

# EVOLVE-BLOCK-END
