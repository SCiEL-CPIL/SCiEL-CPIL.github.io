"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csr_matrix


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * z_i * z_ip1
    
    # X terms: -field * sum X_i
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                idx1 = idx | (1 << i)
                H[idx, idx1] += -field
                H[idx1, idx] += -field
    
    return H


def _apply_circuit_fast(thetas, n, layers):
    """
    Apply the circuit to initial |0...0> state efficiently.
    Returns the final state vector.
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Vectorized: for each pair of basis states differing only in bit i
            mask = np.arange(dim)
            idx_0 = mask[(mask >> i) & 1 == 0]
            idx_1 = idx_0 | (1 << i)
            a = state[idx_0].copy()
            b = state[idx_1].copy()
            state[idx_0] = c * a - s * b
            state[idx_1] = s * a + c * b
        
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask = np.arange(dim)
                both_one = ((mask >> i) & 1) & ((mask >> (i + 1)) & 1)
                state[both_one == 1] *= -1.0
    
    return state


def _circuit_energy(thetas, n, coupling, field, layers, H):
    """Compute energy expectation <psi|H|psi> for the circuit state."""
    state = _apply_circuit_fast(thetas, n, layers)
    H_state = H @ state
    energy = np.real(np.vdot(state, H_state))
    return energy


def _circuit_energy_grad(thetas, n, coupling, field, layers, H):
    """
    Compute energy and gradient using finite differences.
    Returns (energy, gradient_array).
    """
    state = _apply_circuit_fast(thetas, n, layers)
    H_state = H @ state
    energy = np.real(np.vdot(state, H_state))
    
    # Compute gradient via central differences
    eps = 1e-6
    grad = np.zeros_like(thetas, dtype=np.float64)
    flat_thetas = thetas.flatten()
    flat_grad = np.zeros_like(flat_thetas)
    
    for k in range(len(flat_thetas)):
        thetas_plus = thetas.copy()
        thetas_plus.flatten()[k] += eps
        e_plus = _circuit_energy(thetas_plus, n, coupling, field, layers, H)
        
        thetas_minus = thetas.copy()
        thetas_minus.flatten()[k] -= eps
        e_minus = _circuit_energy(thetas_minus, n, coupling, field, layers, H)
        
        flat_grad[k] = (e_plus - e_minus) / (2 * eps)
    
    return energy, flat_grad.reshape(thetas.shape)


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _mean_field_init(n, coupling, field, layers):
    """
    Generate diverse initializations based on mean-field physics.
    For transverse field Ising model:
    - Strong coupling (large coupling/field): favor Z-aligned states
    - Strong field: favor X-aligned states (|+>)
    """
    inits = []
    
    # Product |+> state (all pi/2)
    inits.append(np.full((layers, n), np.pi / 2.0))
    
    # Product |0> state (all 0)
    inits.append(np.zeros((layers, n)))
    
    # Alternating patterns based on physics
    ratio = coupling / max(field, 1e-10)
    
    # If coupling dominates, favor aligned Z
    if ratio > 1.0:
        # Favor |0> or |1> aligned
        inits.append(np.zeros((layers, n)))
        inits.append(np.full((layers, n), np.pi))  # |1>
        # Small deviations from 0
        for delta in [0.2, 0.4, 0.6]:
            inits.append(np.full((layers, n), delta))
            inits.append(np.full((layers, n), np.pi - delta))
    else:
        # Field dominates, favor |+>
        inits.append(np.full((layers, n), np.pi / 2.0))
        # Deviations from pi/2
        for delta in [0.2, 0.4, 0.6]:
            inits.append(np.full((layers, n), np.pi / 2.0 + delta))
            inits.append(np.full((layers, n), np.pi / 2.0 - delta))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        inits.append(rng.uniform(-np.pi, np.pi, (layers, n)))
    
    # Antiferromagnetic-like patterns
    for phase in [0, np.pi / 4, np.pi / 2]:
        pattern = np.array([phase + i * np.pi / n for i in range(n)])
        inits.append(np.tile(pattern, (layers, 1)))
    
    return inits


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # Precompute Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Objective function
    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Penalty
        thetas = params.reshape(layers, n)
        e = _circuit_energy(thetas, n, coupling, field, layers, H)
        return e
    
    # Generate diverse initializations
    inits = _mean_field_init(n, coupling, field, layers)
    
    # Evaluate all initializations and sort by objective value
    init_scores = []
    for x_init in inits:
        val = obj(x_init.flatten())
        init_scores.append((val, x_init.flatten()))
    
    init_scores.sort(key=lambda x: x[0])
    
    best_params = init_scores[0][1]
    best_val = init_scores[0][0]
    
    # Optimization with multiple restarts from best initializations
    n_restarts = min(8, len(init_scores))
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25.0:
            break
        
        x_start = init_scores[r][1]
        
        try:
            # Phase 1: Coarse optimization with BFGS
            res = minimize(
                obj,
                x_start,
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-5}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
            
            # Phase 2: Fine polish with tighter tolerance
            elapsed = time.time() - start_time
            if elapsed < deadline - 15.0:
                res2 = minimize(
                    obj,
                    res.x,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
                    
        except Exception:
            pass
    
    # Final high-precision polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res_final = minimize(
                obj,
                best_params,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
