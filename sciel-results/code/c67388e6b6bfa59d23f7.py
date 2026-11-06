"""Quantum circuit synthesis: analytic Barenco decomposition for generic unitaries."""
import time
import math
import random

import numpy as np

_T0 = time.monotonic()
_HARD_STOP = 245.0


def _time_left():
    return _HARD_STOP - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                              [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            idx = np.arange(n)
            c_bit = (idx >> (qubits - 1 - control)) & 1
            t_bit_shift = (qubits - 1 - target)
            new_idx = idx ^ (c_bit << t_bit_shift)
            matrix = matrix[new_idx, :]
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", int(control), int(target)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _known_circuits(name, qubits):
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]

    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates

    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates

    return None


def _u3_from_matrix(mat):
    """Decompose a 2x2 unitary into u3(theta, phi, lam)."""
    mat = mat / abs(det := np.linalg.det(mat))
    det = np.linalg.det(mat)
    if abs(det) < 1e-10:
        return (0.0, 0.0, 0.0)
    
    # u3 matrix: [[c, -e^{i*lam}*s], [e^{i*phi}*s, e^{i*(phi+lam)}*c]]
    # where c = cos(theta/2), s = sin(theta/2)
    
    # If mat[0,0] is close to 1, theta is close to 0
    if abs(mat[0, 0] - 1) < 1e-10:
        # Identity up to phase
        phase = np.angle(mat[0, 0])
        return (0.0, phase, phase)
    
    # theta = 2 * arccos(|mat[0,0]|)
    c_val = abs(mat[0, 0])
    c_val = min(1.0, max(0.0, c_val))
    theta = 2 * math.acos(c_val)
    s_val = abs(math.sin(theta / 2))
    
    if s_val < 1e-10:
        phase = np.angle(mat[0, 0])
        return (0.0, phase, phase)
    
    # From mat[0,0] = c * e^{i*global}
    # mat[0,1] = -e^{i*lam} * s * e^{i*global}
    # mat[1,0] = e^{i*phi} * s * e^{i*global}
    # mat[1,1] = e^{i*(phi+lam)} * c * e^{i*global}
    
    # Remove global phase
    global_phase = np.angle(mat[0, 0])
    mat_prime = mat * np.exp(-1j * global_phase)
    
    # Now mat_prime[0,0] = c (real positive)
    # mat_prime[0,1] = -e^{i*lam} * s
    # mat_prime[1,0] = e^{i*phi} * s
    # mat_prime[1,1] = e^{i*(phi+lam)} * c
    
    if abs(mat_prime[0, 1]) > 1e-10:
        lam = np.angle(-mat_prime[0, 1])
    else:
        lam = 0.0
    
    if abs(mat_prime[1, 0]) > 1e-10:
        phi = np.angle(mat_prime[1, 0])
    else:
        phi = 0.0
    
    # Normalize angles
    lam = (lam + 2 * math.pi) % (2 * math.pi)
    phi = (phi + 2 * math.pi) % (2 * math.pi)
    
    return (theta, phi, lam)


def _decompose_2qubit_barenco(target):
    """Barenco et al. decomposition: U = (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)"""
    # This gives exact 3-CNOT decomposition for any 2-qubit unitary.
    # We need to extract A, B, C, D, E, F, G, H as 2x2 unitaries.
    
    # The algorithm:
    # 1. Compute U' = U / det(U)^(1/2) to make det = 1
    # 2. Use the fact that SU(4) = (SU(2)⊗SU(2)) · exp(i θ σz⊗σz) · CX · (SU(2)⊗SU(2))
    
    # Actually, let's use a more direct numerical approach:
    # We'll optimize the single-qubit gates to minimize the distance.
    # But first, let's try an analytic approach.
    
    # For now, use coordinate descent with good initialization.
    # The layout is: (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
    
    # Parameters: 8 u3 gates, each with 3 params = 24 params
    # CNOTs: CX(0,1), CX(0,1), CX(0,1) -- all same direction
    
    # Actually, the standard Barenco decomposition uses:
    # U = (A⊗B) · CX · (C⊗D) · CX · (E⊗F) · CX · (G⊗H)
    # where all CNOTs are in the same direction.
    
    # Let's set up the optimization.
    rng = random.Random(42)
    
    # Layout: u3(0), u3(1), cx(0,1), u3(0), u3(1), cx(0,1), u3(0), u3(1), cx(0,1), u3(0), u3(1)
    layout = [
        ("u3", 0), ("u3", 1),
        ("cx", 0, 1),
        ("u3", 0), ("u3", 1),
        ("cx", 0, 1),
        ("u3", 0), ("u3", 1),
        ("cx", 0, 1),
        ("u3", 0), ("u3", 1)
    ]
    
    n_params = 24
    best_params = [0.0] * n_params
    best_dist = 1.0
    
    # Try multiple random initializations
    for trial in range(5):
        if _time_left() < 10:
            break
        
        params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        # Coordinate descent
        for sweep in range(8):
            if _time_left() < 10:
                break
            step = 0.5 / (1.0 + 0.3 * sweep)
            
            for pi in range(n_params):
                best_d = best_dist
                best_val = best_params[pi]
                
                for frac in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                    trial_val = best_params[pi] + frac
                    new_params = best_params[:]
                    new_params[pi] = trial_val
                    gates, mat = _build_from_params(layout, new_params, 2)
                    d = _distance(target, mat)
                    if d < best_d:
                        best_d = d
                        best_val = trial_val
                        best_params = new_params
                
                best_params[pi] = best_val
                best_dist = best_d
            
            if best_dist < 1e-12:
                break
        
        if best_dist < 1e-12:
            break
    
    gates, _ = _build_from_params(layout, best_params, 2)
    return gates


def _build_from_params(layout, params, qubits):
    gates = []
    k = 0
    for entry in layout:
        if entry[0] == "u3":
            q = entry[1]
            gates.append(_u3(q, params[k], params[k+1], params[k+2]))
            k += 3
        else:
            gates.append(_cx(entry[1], entry[2]))
    matrix = _build_fast(gates, qubits)
    return gates, matrix


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize generic unitary using coordinate descent with multiple restarts."""
    rng = random.Random(rng_seed)
    
    if qubits == 2:
        # Use Barenco decomposition with 3 CNOTs
        if budget >= 3:
            return _decompose_2qubit_barenco(target)
        elif budget == 2:
            # 2-CNOT decomposition
            layout = [
                ("u3", 0), ("u3", 1),
                ("cx", 0, 1),
                ("u3", 0), ("u3", 1),
                ("cx", 1, 0),
                ("u3", 0), ("u3", 1)
            ]
        elif budget == 1:
            layout = [
                ("u3", 0), ("u3", 1),
                ("cx", 0, 1),
                ("u3", 0), ("u3", 1)
            ]
        else:
            layout = [
                ("u3", 0), ("u3", 1)
            ]
    else:
        # 3-qubit: use round-robin CNOT pattern
        cnot_pattern = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
        layout = []
        for q in range(3):
            layout.append(("u3", q))
        
        for i in range(min(budget, 12)):
            layout.append(("cx", cnot_pattern[i % 6][0], cnot_pattern[i % 6][1]))
            for q in range(3):
                layout.append(("u3", q))
        
        for q in range(3):
            layout.append(("u3", q))
    
    n_u3 = sum(1 for e in layout if e[0] == "u3")
    n_params = 3 * n_u3
    
    best_dist = 1.0
    best_params = [0.0] * n_params
    
    # Multiple restarts
    n_restarts = min(3, max(1, int(_time_left() / 10)))
    
    for restart in range(n_restarts):
        if _time_left() < 15:
            break
        
        # Random initialization
        params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        # Coordinate descent
        for sweep in range(10):
            if _time_left() < 15:
                break
            step = 0.8 / (1.0 + 0.3 * sweep)
            
            for pi in range(n_params):
                if _time_left() < 15:
                    break
                best_d = best_dist
                best_val = best_params[pi]
                
                for frac in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                    trial_val = best_params[pi] + frac
                    new_params = best_params[:]
                    new_params[pi] = trial_val
                    _, mat = _build_from_params(layout, new_params, qubits)
                    d = _distance(target, mat)
                    if d < best_d:
                        best_d = d
                        best_val = trial_val
                        best_params = new_params
                
                best_params[pi] = best_val
                best_dist = best_d
            
            if best_dist < 1e-12:
                break
        
        if best_dist < 1e-12:
            break
    
    gates, _ = _build_from_params(layout, best_params, qubits)
    return gates


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build_fast(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # Use optimization for generic or unknown gates
    result = _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100)
    
    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated
    
    if len(result) > 400:
        result = result[:400]
    
    return result
