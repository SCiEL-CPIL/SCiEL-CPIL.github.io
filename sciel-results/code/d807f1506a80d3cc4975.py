import numpy as np
import time
from scipy.optimize import minimize
import sys

# Cache for optimized parameters
_PARAM_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (0-indexed, LSB is qubit 0).
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Interaction terms: -coupling * Z_i Z_{i+1}
    # Z = diag(1, -1)
    # We can construct Z_i Z_{i+1} by iterating over basis states
    # Or use Kronecker products. For small n, iterating is fine and avoids large intermediate matrices.
    for state in range(dim):
        # Get bits
        bits = [(state >> i) & 1 for i in range(n)]
        # Z eigenvalue is 1 if bit 0, -1 if bit 1
        # Z_i Z_{i+1} eigenvalue is z_i * z_{i+1}
        # Sum over i=0 to n-2
        zz_val = 0.0
        for i in range(n - 1):
            z_i = 1.0 if bits[i] == 0 else -1.0
            z_ip1 = 1.0 if bits[i+1] == 0 else -1.0
            zz_val += z_i * z_ip1
        H[state, state] += -coupling * zz_val

    # Field terms: -field * X_i
    # X flips bit i
    for i in range(n):
        for state in range(dim):
            # Flip bit i
            new_state = state ^ (1 << i)
            H[state, new_state] += -field

    return H

def _simulate_circuit(theta, n):
    """
    Simulate the circuit and return the final state vector.
    theta: shape (layers, n)
    n: number of qubits
    """
    layers = theta.shape[0]
    dim = 2 ** n
    # Start state |0>^n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # Precompute rotation matrices? They depend on theta.
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to all qubits i
        # Reshape psi to (2, 2, ..., 2)
        # Axis i corresponds to qubit i.
        # In numpy reshape with C-order, the last axis is the fastest changing (LSB).
        # So axis 0 is qubit 0 (LSB), axis 1 is qubit 1, etc.
        psi_reshaped = psi.reshape([2] * n)
        
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            # Ry = [[c, -s], [s, c]]
            # We need to apply this to axis i.
            # Extract the 2x2 matrix slice?
            # Easier: use np.einsum or manual index manipulation.
            # Let's use a helper to apply single-qubit gate.
            
            # Move axis i to the front
            psi_moved = np.moveaxis(psi_reshaped, i, 0)
            # Apply matrix: new[0, ...] = c*old[0, ...] - s*old[1, ...]
            # new[1, ...] = s*old[0, ...] + c*old[1, ...]
            # Be careful with in-place modification if possible, but copying is safer for correctness first.
            # Since n is small, copying is fine.
            
            psi_new = np.empty_like(psi_moved)
            psi_new[0] = c * psi_moved[0] - s * psi_moved[1]
            psi_new[1] = s * psi_moved[0] + c * psi_moved[1]
            
            # Move axis back
            psi_reshaped = np.moveaxis(psi_new, 0, i)
            
        # If not the last layer, apply CZ gates
        if l < layers - 1:
            # Apply CZ on (i, i+1) for i in 0..n-2 in ascending order
            for i in range(n - 1):
                # CZ is identity except when both bits are 1, where it is -1.
                # We need to multiply psi[state] by -1 if bit i and i+1 are 1.
                # Iterate over states? Or use masks.
                # Mask approach:
                # Create a mask of size dim where mask[state] is True if bits i and i+1 are 1.
                # For n<=6, dim<=64, iterating is very fast.
                for state in range(dim):
                    if ((state >> i) & 1) and ((state >> (i+1)) & 1):
                        psi[state] *= -1.0

    return psi

def _energy_gap(theta, n, H, E0):
    """
    Compute the energy gap for a given set of parameters.
    """
    psi = _simulate_circuit(theta, n)
    # Energy = psi^dagger H psi
    Hpsi = H @ psi
    E = np.vdot(psi, Hpsi).real
    return E - E0

def angles(n, coupling, field, layers):
    key = (n, coupling, field)
    
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Build Hamiltonian and get ground state
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    # Ground state vector is eigvecs[:, 0]
    
    best_theta = np.zeros((layers, n))
    best_energy = np.inf
    
    # Multi-start optimization
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        theta_init = rng.uniform(0, 2 * np.pi, size=(layers, n))
        
        # Flatten for optimizer
        x0 = theta_init.flatten()
        
        def objective(x):
            theta = x.reshape((layers, n))
            return _energy_gap(theta, n, H, E0)
        
        # L-BFGS-B
        # Bounds: [0, 2*pi]
        bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
            
    _PARAM_CACHE[key] = best_theta
    return best_theta
