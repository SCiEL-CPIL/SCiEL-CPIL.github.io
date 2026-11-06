"""Optimized shallow-circuit Ising ground-state ansatz."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_H(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms
    for i in range(n - 1):
        for state in range(dim):
            b_i = (state >> i) & 1
            b_j = (state >> (i + 1)) & 1
            z_i = 1 - 2 * b_i
            z_j = 1 - 2 * b_j
            H[state, state] += -coupling * z_i * z_j
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground-state energy via full diagonalization."""
    H = _build_H(n, coupling, field)
    # Use eigvalsh for Hermitian matrices (real symmetric here)
    evals = np.linalg.eigvalsh(H)
    return evals[0]


def _circuit_state(n, theta, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    theta has shape (layers, n).
    Each layer: Ry(theta[l, i]) on qubit i for all i.
    Between layers: CZ on adjacent pairs (i, i+1) in ascending order.
    No entangler after final layer.
    """
    dim = 1 << n
    # Start in |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            # Ry = [[c, -s], [s, c]] in the subspace where qubit i is 0/1
            # We need to apply this to every pair of basis states differing in bit i
            new_state = np.zeros_like(state)
            for state_idx in range(dim):
                if (state_idx >> i) & 1 == 0:
                    # This state has bit i = 0
                    partner = state_idx | (1 << i)  # bit i = 1
                    new_state[state_idx] += c * state[state_idx]
                    new_state[partner] += s * state[state_idx]
                else:
                    # This state has bit i = 1
                    partner = state_idx & ~(1 << i)  # bit i = 0
                    new_state[partner] += s * state[state_idx]
                    new_state[state_idx] += c * state[state_idx]
            state = new_state

        # Apply CZ between layers (not after last)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1: flip phase of |11>
                for state_idx in range(dim):
                    b_i = (state_idx >> i) & 1
                    b_j = (state_idx >> (i + 1)) & 1
                    if b_i == 1 and b_j == 1:
                        state[state_idx] *= -1

    return state


def _circuit_energy(n, coupling, field, theta, layers):
    """Compute <psi|H|psi> for the circuit state."""
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, theta, layers)
    return np.real(np.vdot(psi, H @ psi))


def _objective(params, n, coupling, field, layers):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    E_circuit = _circuit_energy(n, coupling, field, theta, layers)
    E_exact = _exact_ground_energy(n, coupling, field)
    gap = E_circuit - E_exact
    return gap / n


def _initial_guess(n, field, layers):
    """
    Heuristic initial guess based on physics.
    For transverse-field Ising model, the ground state has magnetization
    along X direction at high field, and along Z at low field.
    """
    theta = np.zeros((layers, n))
    # At high field, ground state is close to |+>^n, so Ry(pi/2) makes |+>
    # At low field, ground state has more Z-magnetization
    # Interpolate: at field=0, we want |0> or |1> depending on coupling sign
    # For coupling=1 (ferromagnetic), ground state at field=0 is (|00..0> + |11..1>)/sqrt(2)
    # This is hard to reach with shallow circuits.
    
    # Simple heuristic: use field-dependent angle
    # At field=1.65, close to |+>, so theta ~ pi/2
    # At field=0.45, more Z-like, so theta smaller
    base_angle = np.arctan(field / coupling)  # crude
    base_angle = np.clip(base_angle, 0.1, np.pi / 2)
    
    for l in range(layers):
        # Spread angles across layers
        scale = 1.0 / layers
        theta[l, :] = base_angle * scale
    
    # Ensure final layer can add remaining rotation
    remaining = base_angle - theta[:layers-1].sum(axis=0)[0] if layers > 1 else base_angle
    if layers > 1:
        theta[layers-1, :] = np.clip(remaining, -np.pi/2, np.pi/2)
    
    return theta


def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to approximate ground state.
    Returns theta of shape (layers, n).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Get exact ground energy
    E_exact = _exact_ground_energy(n, coupling, field)
    
    # Initial guess
    theta0 = _initial_guess(n, field, layers)
    x0 = theta0.flatten()
    
    best_theta = theta0.copy()
    best_score = _objective(x0, n, coupling, field, layers)
    
    # Try multiple starting points and optimizations
    starts = [x0]
    
    # Add a few heuristic starts
    # All pi/2 (maximally entangled-ish)
    starts.append(np.full((layers, n), np.pi / 2).flatten())
    # All 0
    starts.append(np.zeros((layers, n)))
    # Gradient-like: first layer small, last layer large
    for frac in [0.3, 0.5, 0.7]:
        t = np.zeros((layers, n))
        for l in range(layers):
            t[l, :] = frac * np.pi / 2 * (l + 1) / layers
        starts.append(t.flatten())
    
    for x_init in starts:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # If time remains, try Nelder-Mead from best point
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, layers),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_theta = result2.x.reshape(layers, n)
        except Exception:
            pass
    
    return best_theta
# EVOLVE-BLOCK-END
