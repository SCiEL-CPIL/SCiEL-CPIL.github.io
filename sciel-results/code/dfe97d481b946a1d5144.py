"""Combined approach: exact known gates + KAK-style structured initialization +
Nelder-Mead simplex with coordinate-descent polish for generic unitaries.
Uses fast in-place matrix application for CNOTs and kron-based u3 application."""
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]], dtype=complex)


def _apply_u3_to_matrix(matrix, q, theta, phi, lam, qubits):
    n = 2 ** qubits
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    bit_pos = qubits - 1 - q
    shift = 1 << bit_pos
    new_matrix = np.zeros_like(matrix)
    for i in range(n):
        if (i & shift) == 0:
            j = i | shift
            new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
            new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
    return new_matrix


def _apply_cx_to_matrix(matrix, control, target_q, qubits):
    n = 2 ** qubits
    cshift = 1 << (qubits - 1 - control)
    tshift = 1 << (qubits - 1 - target_q)
    perm = np.arange(n)
    for i in range(n):
        if (i & cshift):
            perm[i] = i ^ tshift
    return matrix[perm, :]


def _build_circuit_matrix(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3_to_matrix(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target_q = gate
            matrix = _apply_cx_to_matrix(matrix, control, target_q, qubits)
    return matrix


def _distance(gates, qubits, target):
    n = 2 ** qubits
    matrix = _build_circuit_matrix(gates, qubits)
    tr_val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr_val) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _exact_cnot():
    return [_cx(0, 1)]


def _exact_iswap():
    gates = [
        _u3(0, math.pi / 2, 0, math.pi),
        _u3(1, math.pi / 2, 0, math.pi),
        _cx(0, 1),
        _u3(0, 0, 0, -math.pi / 2),
        _u3(1, 0, 0, -math.pi / 2),
        _cx(0, 1),
        _u3(0, math.pi / 2, 0, math.pi),
        _u3(1, math.pi / 2, 0, math.pi),
    ]
    return gates


def _exact_swap():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_toffoli():
    gates = [
        _u3(2, math.pi / 2, 0, math.pi),
        _u3(2, 0, 0, math.pi / 4),
        _cx(1, 2),
        _u3(1, 0, 0, -math.pi / 4),
        _u3(2, 0, 0, math.pi / 4),
        _cx(0, 2),
        _u3(0, 0, 0, -math.pi / 4),
        _u3(1, 0, 0, math.pi / 4),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi / 4),
        _u3(0, 0, 0, math.pi / 4),
        _u3(1, 0, 0, -math.pi / 4),
        _cx(0, 2),
        _u3(2, math.pi / 2, 0, math.pi),
    ]
    return gates


def _try_known_gates(problem, target, qubits, budget):
    if qubits == 2:
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_target, atol=1e-8):
            return _exact_cnot()
        
        iswap_target = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex)
        if np.allclose(target, iswap_target, atol=1e-8):
            return _exact_iswap()
        
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap_target, atol=1e-8):
            return _exact_swap()
    
    if qubits == 3:
        toffoli_target = np.eye(8, dtype=complex)
        toffoli_target[6,6] = 0
        toffoli_target[7,7] = 0
        toffoli_target[6,7] = 1
        toffoli_target[7,6] = 1
        if np.allclose(target, toffoli_target, atol=1e-8):
            return _exact_toffoli()
    
    return None


def _build_layout(qubits, budget):
    """Build a fixed layout of u3 and cx gates for the given qubits and budget."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    # Initial u3 gates on all qubits
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # CNOT ladder with u3 gates after each CNOT
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    return layout


def _param_map_from_layout(layout):
    """Create a mapping from parameter indices to (gate_index, angle_index)."""
    param_map = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0))  # theta
            param_map.append((i, 1))  # phi
            param_map.append((i, 2))  # lam
    return param_map


def _get_circuit_from_params(layout, param_map, params):
    """Build a circuit from the layout and parameter values."""
    gates = [list(g) for g in layout]
    for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
        gates[gate_idx][2 + angle_idx] = params[p_idx]
    return gates


def _init_params_structured(target, qubits, budget, layout, param_map):
    """Initialize parameters using diagonal-phase extraction and structured heuristics."""
    n_params = len(param_map)
    rng = random.Random(7777 + qubits * 100 + budget)
    
    params = [0.0] * n_params
    
    # Extract diagonal phases from target
    # For a circuit that is close to diagonal, we can extract phases
    # u3(q, 0, phi, lam) = diag(e^{i*phi/2}, e^{i*(phi+lam)/2})
    
    # For each u3 gate in the layout, try to match its contribution
    # to the diagonal of the target
    n = 2 ** qubits
    
    # Try to extract phases from the diagonal of the target
    # This is a rough heuristic: for each qubit, look at the diagonal entries
    # and try to infer phases
    
    for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
        gate = layout[gate_idx]
        q = gate[1]
        bit_pos = qubits - 1 - q
        shift = 1 << bit_pos
        
        # Look at diagonal entries where this qubit is 0 vs 1
        diag_0 = []
        diag_1 = []
        for i in range(n):
            if (i & shift) == 0:
                diag_0.append(target[i, i])
            else:
                diag_1.append(target[i, i])
        
        if diag_0 and diag_1:
            # Average phases
            phase_0 = np.angle(np.mean(diag_0))
            phase_1 = np.angle(np.mean(diag_1))
            
            # u3 with theta=0: diag(e^{i*phi/2}, e^{i*(phi+lam)/2})
            # So phi/2 = phase_0, (phi+lam)/2 = phase_1
            # phi = 2*phase_0, lam = 2*phase_1 - phi
            phi_val = 2 * phase_0
            lam_val = 2 * phase_1 - phi_val
            
            # Map to parameter index
            if angle_idx == 0:  # theta
                params[p_idx] = rng.uniform(-0.3, 0.3)
            elif angle_idx == 1:  # phi
                params[p_idx] = phi_val % (2 * math.pi)
            else:  # lam
                params[p_idx] = lam_val % (2 * math.pi)
        else:
            params[p_idx] = rng.uniform(-0.5, 0.5)
    
    return params


def _nelder_mead(objective, x0, max_iter=3000, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead simplex optimization with deadline awareness."""
    n = len(x0)
    
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        new_point = list(x0)
        step = 0.5 if abs(x0[i]) < 1e-6 else 0.5 * abs(x0[i])
        new_point[i] += step
        simplex.append(new_point)
    
    # Evaluate initial simplex
    values = [objective(point) for point in simplex]
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        # Sort by value
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        x_r = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        f_r = objective(x_r)
        
        if values[0] <= f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        elif f_r < values[0]:
            # Expansion
            x_e = [centroid[j] + gamma * (x_r[j] - centroid[j]) for j in range(n)]
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        else:
            # Contraction
            x_c = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            f_c = objective(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


def _coordinate_descent_polish(objective, x0, max_sweeps=15, initial_step=0.5):
    """Coordinate descent polish with adaptive step size."""
    n = len(x0)
    best = list(x0)
    best_val = objective(best)
    
    rng = random.Random(9999)
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        
        step = initial_step / (1.0 + sweep * 0.3)
        improved = False
        
        for i in range(n):
            if _check_deadline():
                break
            
            base = best[i]
            trials = [base + step, base - step, base + step * 0.5, base - step * 0.5]
            if sweep < 3:
                trials.append(base + rng.uniform(-math.pi, math.pi))
            
            for trial in trials:
                values = best[:]
                values[i] = trial
                val = objective(values)
                if val < best_val:
                    best = values
                    best_val = val
                    improved = True
        
        if not improved and sweep > 5:
            break
    
    return best, best_val


def _optimize_generic(target, qubits, budget):
    """Optimize using structured initialization + Nelder-Mead + coordinate descent polish."""
    layout = _build_layout(qubits, budget)
    param_map = _param_map_from_layout(layout)
    n_params = len(param_map)
    
    def objective(params):
        if _check_deadline():
            return 1.0
        gates = _get_circuit_from_params(layout, param_map, params)
        return _distance(gates, qubits, target)
    
    # Initialize parameters with structured heuristic
    best_params = _init_params_structured(target, qubits, budget, layout, param_map)
    best_dist = objective(best_params)
    
    # Run Nelder-Mead
    if not _check_deadline():
        nm_params, nm_dist = _nelder_mead(objective, best_params, max_iter=2500)
        if nm_dist < best_dist:
            best_params, best_dist = nm_params, nm_dist
    
    # Coordinate descent polish
    if not _check_deadline():
        cd_params, cd_dist = _coordinate_descent_polish(objective, best_params, max_sweeps=15, initial_step=0.5)
        if cd_dist < best_dist:
            best_params, best_dist = cd_params, cd_dist
    
    # Try a few random restarts if still far from target
    rng = random.Random(4242 + qubits * 1000 + budget)
    if best_dist > 0.01 and not _check_deadline():
        for restart in range(5):
            if _check_deadline():
                break
            trial_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            trial_dist = objective(trial_params)
            
            if trial_dist < best_dist:
                best_params = trial_params
                best_dist = trial_dist
                # Polish with Nelder-Mead
                if not _check_deadline():
                    nm_params, nm_dist = _nelder_mead(objective, best_params, max_iter=1000)
                    if nm_dist < best_dist:
                        best_params, best_dist = nm_params, nm_dist
                # Coordinate descent polish
                if not _check_deadline():
                    cd_params, cd_dist = _coordinate_descent_polish(objective, best_params, max_sweeps=10, initial_step=0.3)
                    if cd_dist < best_dist:
                        best_params, best_dist = cd_params, cd_dist
    
    # Final polish with very small steps if needed
    if best_dist > 0.001 and not _check_deadline():
        for sweep in range(5):
            if _check_deadline():
                break
            step = 0.01 / (1 + sweep)
            for i in range(n_params):
                if _check_deadline():
                    break
                for trial in (best_params[i] + step, best_params[i] - step):
                    values = best_params[:]
                    values[i] = trial
                    value = objective(values)
                    if value < best_dist:
                        best_params, best_dist = values, value
    
    return _get_circuit_from_params(layout, param_map, best_params)


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    known_circuit = _try_known_gates(problem, target, qubits, budget)
    if known_circuit is not None:
        cnot_count = sum(1 for g in known_circuit if g[0] == "cx")
        if cnot_count <= budget and len(known_circuit) <= 400:
            return known_circuit
    
    return _optimize_generic(target, qubits, budget)
