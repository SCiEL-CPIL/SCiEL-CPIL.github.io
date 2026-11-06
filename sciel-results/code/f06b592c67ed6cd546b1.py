"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 260.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _nelder_mead(f, x0, max_iter=200, tol=1e-10, deadline=None):
    """Simple Nelder-Mead optimizer with deadline guard."""
    n = len(x0)
    alpha, gamma, rho = 1.0, 2.0, 0.5
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = x0[i] + 0.1 * (abs(x0[i]) + 1.0)
        simplex.append(point)
    
    fvals = [f(p) for p in simplex]
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        worst = simplex[-1]
        reflected = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        f_reflected = f(reflected)
        
        if fvals[0] <= f_reflected < fvals[-2]:
            simplex[-1] = reflected
            fvals[-1] = f_reflected
        elif f_reflected < fvals[0]:
            # Expansion
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            f_expanded = f(expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                fvals[-1] = f_expanded
            else:
                simplex[-1] = reflected
                fvals[-1] = f_reflected
        else:
            # Contraction
            contracted = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
            f_contracted = f(contracted)
            if f_contracted < fvals[-1]:
                simplex[-1] = contracted
                fvals[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = f(simplex[i])
    
    return simplex[0], fvals[0]


def synthesize(problem):
    """Adaptive CNOT layout with Nelder-Mead optimization and deadline guard."""
    start_time = time.time()
    deadline = start_time + (_DEADLINE - (time.time() - start_time))
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    rng = random.Random(4242 + qubits)
    
    # Generate multiple candidate CNOT layouts
    def generate_layout(seed, num_cnots):
        pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
        layout = []
        for layer in range(num_cnots):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", *pairs[(seed + layer) % len(pairs)]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        return layout
    
    # Try a few different CNOT orderings
    best_gates = None
    best_distance = float('inf')
    
    num_candidates = min(5, max(1, budget))
    for cand_idx in range(num_candidates):
        if time.time() > deadline - 10:  # Stop if we're close to deadline
            break
        
        layout = generate_layout(cand_idx, budget)
        num_u3 = sum(1 for g in layout if g[0] == "u3")
        dim = 3 * num_u3
        
        def circuit(values):
            gates, k = [], 0
            for gate in layout:
                if gate[0] == "u3":
                    gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                    k += 3
                else:
                    gates.append(list(gate))
            return gates
        
        # Objective function
        def objective(values):
            return _distance(target, _build(circuit(values), qubits))
        
        # Initial guess: all zeros
        x0 = [0.0] * dim
        
        # Run Nelder-Mead with deadline guard
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=100, tol=1e-8, deadline=deadline - 5)
        
        if f_opt < best_distance:
            best_distance = f_opt
            best_gates = circuit(x_opt)
        
        # If we're already very good, stop early
        if best_distance < 1e-8:
            break
    
    # If no good result, fall back to all-zeros circuit
    if best_gates is None:
        layout = generate_layout(0, budget)
        best_gates = []
        for gate in layout:
            if gate[0] == "u3":
                best_gates.append(["u3", gate[1], 0.0, 0.0, 0.0])
            else:
                best_gates.append(list(gate))
    
    return best_gates
# EVOLVE-BLOCK-END
