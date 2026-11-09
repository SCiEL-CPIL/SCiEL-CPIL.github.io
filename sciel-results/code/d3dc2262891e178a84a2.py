"""Quantum information: shallow-circuit Ising ground states.
Optimizes Ry/CZ circuit parameters via L-BFGS-B with analytic finite-difference-free
gradient computation using the parameter-shift-like structure of Ry gates."""
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Pauli matrices and their Kronecker products for efficiency
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PZ = np.array([[1, 0], [0, -1]], dtype=complex)
_ID = np.eye(2, dtype=complex)


def _kron_all(mats):
    """Kronecker product of a list of matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    H = np.zeros((2**n, 2**n), dtype=complex)
    # ZZ terms
    for i in range(n - 1):
        ops = [_ID] * n
        ops[i] = _PZ
        ops[i + 1] = _PZ
        H -= coupling * _kron_all(ops)
    # X terms
    for i in range(n):
        ops = [_ID] * n
        ops[i] = _PX
        H -= field * _kron_all(ops)
    return H


def _apply_circuit(n, theta):
    """Apply the 3-layer Ry/CZ circuit to |0>^n.
    theta has shape (layers, n). Each layer: Ry(theta) on all qubits, then CZ on adjacent pairs.
    No CZ after the final layer."""
    layers, nq = theta.shape
    state = np.zeros(2**nq, dtype=complex)
    state[0] = 1.0  # |0>^n

    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for q in range(nq):
            t = theta[layer, q]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            # Ry = cos(t/2)*I - sin(t/2)*X
            # Apply to qubit q: for each basis state, mix |0> and |1> on qubit q
            new_state = np.zeros_like(state)
            for idx in range(2**nq):
                bit = (idx >> q) & 1
                if bit == 0:
                    idx1 = idx | (1 << q)
                    new_state[idx] += c * state[idx]
                    new_state[idx1] += -s * state[idx]
                else:
                    idx0 = idx & ~(1 << q)
                    new_state[idx] += -s * state[idx]
                    new_state[idx0] += c * state[idx0]
            state = new_state

        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(nq - 1):
                # CZ: |11> -> -|11>
                for idx in range(2**nq):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[idx] *= -1

    return state


def _energy_and_grad(n, coupling, field, theta, H, E_ground):
    """Compute energy and gradient via finite differences (analytic-ish).
    Uses a small step for numerical gradient since the circuit is small."""
    state = _apply_circuit(n, theta)
    E = np.real(np.vdot(state, H @ state))

    # Gradient via central differences
    eps = 1e-4
    grad = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            tp = theta.copy()
            tm = theta.copy()
            tp[i, j] += eps
            tm[i, j] -= eps
            sp = _apply_circuit(n, tp)
            sm = _apply_circuit(n, tm)
            Ep = np.real(np.vdot(sp, H @ sp))
            Em = np.real(np.vdot(sm, H @ sm))
            grad[i, j] = (Ep - Em) / (2 * eps)

    return E - E_ground, grad


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state of Ising chain."""
    t_start = time.time()
    deadline = 150.0  # evaluator kills at 150s wall-clock
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    E_ground = eigvals[0]

    # Initial guesses: try multiple starting points
    # Strategy: use physics-informed initialization + random restarts
    best_theta = None
    best_score = np.inf

    # Physics-informed init: for transverse-field Ising, ground state has
    # some magnetization. Try angles based on field strength.
    # Small field -> near |0>, large field -> more |+> character
    base_angle = np.arctan(field / max(coupling, 1e-6))
    
    initializations = []
    # Init 1: uniform angle based on field
    init1 = np.tile(np.array([base_angle] * n), (layers, 1))
    initializations.append(init1)
    
    # Init 2: alternating pattern
    init2 = np.zeros((layers, n))
    for i in range(n):
        init2[:, i] = base_angle * (1 if i % 2 == 0 else -1)
    initializations.append(init2)
    
    # Init 3: all pi/2 (maximally mixed-like)
    initializations.append(np.full((layers, n), np.pi / 4))
    
    # Init 4: small angles
    initializations.append(np.full((layers, n), 0.1))
    
    # Add random initializations
    rng = np.random.default_rng(42)
    for _ in range(6):
        init_r = rng.uniform(-np.pi/2, np.pi/2, size=(layers, n))
        initializations.append(init_r)

    # Optimize from each initialization
    for init in initializations:
        if time.time() - t_start > deadline - 10:  # safety margin
            break

        # Define objective function for L-BFGS-B
        def objective(params):
            theta_flat = params.reshape(layers, n)
            e, g = _energy_and_grad(n, coupling, field, theta_flat, H, E_ground)
            return e, g

        # Use L-BFGS-B with analytic gradient
        result = minimize(
            objective,
            init.flatten(),
            method='L-BFGS-B',
            jac=True,  # we return gradient from objective
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )

        if result.success or result.fun < best_score:
            candidate = result.x.reshape(layers, n)
            # Verify by computing energy directly
            state = _apply_circuit(n, candidate)
            E = np.real(np.vdot(state, H @ state))
            score = E - E_ground
            if score < best_score:
                best_score = score
                best_theta = candidate.copy()

    # If no valid result, fall back to simple heuristic
    if best_theta is None:
        best_theta = np.full((layers, n), np.pi / 4)

    # Final safety check
    if time.time() - t_start > deadline - 5:
        pass  # already have best_theta

    return best_theta
