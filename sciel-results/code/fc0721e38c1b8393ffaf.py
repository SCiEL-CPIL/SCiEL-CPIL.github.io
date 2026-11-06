"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_START_TIME = time.time()
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


def _nelder_mead(f, x0, max_iter=1000, tol=1e-10):
    """Nelder-Mead optimizer."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = x0[i] + 0.1 * (abs(x0[i]) + 1.0)
        simplex.append(point)
    
    fvals = [f(p) for p in simplex]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 2.0
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        # Sort simplex
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if fvals[-1] - fvals[0] < tol:
            break
        
        # Centroid of best n points
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = f(simplex[i])
            else:
                xc = [centroid[j] - rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = f(simplex[i])
        
        # Re-sort
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < tol:
            break
            
    return simplex[0]


def synthesize(problem):
    """Optimize single-qubit rotations for a fixed CNOT layout."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Define a good CNOT layout
    # For 2 qubits, only 1 pair: (0,1)
    # For 3 qubits, we can use a ladder or a specific pattern.
    # A common pattern for 3 qubits is to cycle through pairs: (0,1), (1,2), (0,1), (1,2)...
    # Or just use a fixed sequence that allows full connectivity.
    # Let's use a "ladder" style: 0-1, 1-2, 0-1, 1-2...
    pairs = []
    if qubits == 2:
        pairs = [(0, 1)]
    elif qubits == 3:
        # Cycle through adjacent pairs
        pairs = [(0, 1), (1, 2)]
    else:
        # Fallback for higher qubits (unlikely in this task)
        pairs = [(i, i+1) for i in range(qubits-1)]
        
    layout = []
    num_u3 = 0
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            num_u3 += 1
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        num_u3 += 1
        
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(values):
        return _distance(target, _build(circuit(values), qubits))

    # Initial guess: random
    rng = random.Random(4242 + qubits + budget)
    x0 = [rng.uniform(-math.pi, math.pi) for _ in range(3 * num_u3)]
    
    # Run Nelder-Mead
    best_x = _nelder_mead(objective, x0)
    
    # Check if we hit the deadline, if so, return best found
    if time.time() > _DEADLINE:
        return circuit(best_x)
        
    # If we have time, try a few more restarts with different random seeds
    # But be careful not to exceed the deadline
    current_best_dist = objective(best_x)
    
    # Simple heuristic: if the distance is not very small, try one more restart
    if current_best_dist > 1e-6 and time.time() < _DEADLINE - 5.0:
        x0_2 = [rng.uniform(-math.pi, math.pi) for _ in range(3 * num_u3)]
        x2 = _nelder_mead(objective, x0_2)
        dist2 = objective(x2)
        if dist2 < current_best_dist:
            best_x = x2
            current_best_dist = dist2
            
    return circuit(best_x)
# EVOLVE-BLOCK-END
