"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[state, state] += -coupling * z_i * z_ip1
    
    # X terms: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H


def _apply_circuit(state, angles, n, layers):
    """Apply the Ry/CZ circuit to the initial state |0>^n.
    
    angles: array of shape (layers, n)
    Start with |0>^n, apply Ry(theta) to every qubit, then CZ on adjacent pairs.
    No entangler after the final layer.
    """
    dim = 2 ** n
    # Start with |0>^n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # Ry = [[c, -s], [s, c]]
            new_psi = np.zeros_like(psi)
            for state in range(dim):
                bit = (state >> i) & 1
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    new_psi[state] += c * psi[state]
                    flipped = state ^ (1 << i)
                    new_psi[flipped] += s * psi[state]
                else:
                    # |1> -> -s|0> + c|1>
                    new_psi[state] += c * psi[state]
                    flipped = state ^ (1 << i)
                    new_psi[flipped] += -s * psi[state]
            psi = new_psi
        
        # Apply CZ on adjacent pairs (if not the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                for state in range(dim):
                    bit_i = (state >> i) & 1
                    bit_ip1 = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        psi[state] *= -1
    
    return psi


def _energy(psi, n, coupling, field):
    """Compute the energy of a state for the Ising Hamiltonian."""
    dim = 2 ** n
    # ZZ terms
    energy = 0.0
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            energy += -coupling * z_i * z_ip1 * (psi[state].conj() * psi[state]).real
    
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            energy += -field * (psi[flipped].conj() * psi[state]).real
    
    return energy.real


def _objective(params, n, coupling, field, layers):
    """Objective function: mean energy gap per spin."""
    angles_arr = params.reshape(layers, n)
    psi = _apply_circuit(None, angles_arr, n, layers)
    e_circuit = _energy(psi, n, coupling, field)
    return e_circuit


def _compute_exact_ground(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _init_uniform(n, field, layers):
    """Physics-inspired initialization based on field strength."""
    angles = np.zeros((layers, n))
    # For low field, closer to |+> (theta=pi/2)
    # For high field, closer to |0> (theta=0) with some mixing
    # Interpolate
    base_theta = np.pi / 2 * max(0, 1 - field)
    for layer in range(layers):
        for i in range(n):
            angles[layer, i] = base_theta
    return angles


def _init_from_product_state(n, coupling, field, layers):
    """Initialize by finding best product state approximation."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0]
    
    # Try to find single-qubit rotations that best approximate
    angles = np.zeros((layers, n))
    
    # For each qubit, find the optimal single-qubit rotation
    for i in range(n):
        # Reduce the state to a single qubit by tracing out others
        # For a product state approximation, find the best 1-qubit state
        # This is approximate - use the reduced density matrix
        rho_i = np.zeros((2, 2), dtype=complex)
        for state in range(2 ** n):
            bit = (state >> i) & 1
            for state2 in range(2 ** n):
                bit2 = (state2 >> i) & 1
                # Check if states differ only in qubit i
                if state ^ state2 == (1 << i):
                    rho_i[bit, bit2] += ground_state[state].conj() * ground_state[state2]
    
        # Find optimal theta such that Ry(theta)|0> maximizes overlap
        best_theta = 0
        best_overlap = -1
        for theta in np.linspace(0, np.pi, 50):
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            prod_state = np.array([c, s], dtype=complex)
            overlap = np.abs(prod_state.conj() @ rho_i @ prod_state).real
            if overlap > best_overlap:
                best_overlap = overlap
                best_theta = theta
        
        for layer in range(layers):
            angles[layer, i] = best_theta
    
    return angles


def _random_init(n, layers, scale=1.0):
    """Random initialization."""
    return np.random.uniform(-scale, scale, (layers, n))


def angles(n, coupling, field, layers):
    """Optimize Ry angles for the given Ising chain parameters."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Compute exact ground energy
    e_exact = _compute_exact_ground(n, coupling, field)
    
    def objective_with_gap(params):
        if time.time() > deadline:
            return 1e10
        angles_arr = params.reshape(layers, n)
        psi = _apply_circuit(None, angles_arr, n, layers)
        e_circuit = _energy(psi, n, coupling, field)
        return (e_circuit - e_exact) / n
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Uniform physics-inspired
    inits.append(_init_uniform(n, field, layers))
    
    # 2. Product state inspired
    inits.append(_init_from_product_state(n, coupling, field, layers))
    
    # 3. Zero (|0> state)
    inits.append(np.zeros((layers, n)))
    
    # 4. Pi/2 everywhere (|+> state)
    inits.append(np.full((layers, n), np.pi / 2))
    
    # 5-8. Random inits
    for _ in range(4):
        inits.append(_random_init(n, layers, scale=np.pi / 4))
    
    # Optimize from each initialization
    best_params = None
    best_score = float('inf')
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        params_flat = init.flatten()
        
        # Try multiple optimizers
        for method in ['BFGS', 'Nelder-Mead']:
            if time.time() > deadline - 5:
                break
            
            try:
                result = minimize(
                    objective_with_gap,
                    params_flat,
                    method=method,
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        # Also try L-BFGS-B with bounds
        if time.time() > deadline - 5:
            break
        try:
            bounds = [(-np.pi, np.pi)] * (layers * n)
            result = minimize(
                objective_with_gap,
                params_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 30 and best_params is not None:
        try:
            bounds = [(-np.pi, np.pi)] * (layers * n)
            de_result = differential_evolution(
                objective_with_gap,
                bounds,
                maxiter=10,
                popsize=8,
                tol=1e-6,
                seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Final check: if no valid result, use the best initialization
    if best_params is None:
        best_params = inits[0].flatten()
    
    # Return as (layers, n) array
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
