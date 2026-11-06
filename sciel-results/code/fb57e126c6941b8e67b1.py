"""Combined quantum circuit synthesis program."""
import math
import time
import random
import numpy as np

_DEADLINE = time.time() + 255.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build the unitary matrix from a gate list."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            U = U[idx ^ (bit << (qubits - 1 - t)), :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]


def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]


def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]


def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]


def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]


def _cnot(c, t):
    return ['cx', c, t]


def _decomp_cnot_01():
    return [_cnot(0, 1)]


def _decomp_cnot_10():
    return [_cnot(1, 0)]


def _decomp_swap():
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]


def _decomp_iswap():
    """iSWAP = (H⊗H) · CX(0,1) · (S†⊗S) · CX(0,1) · (H⊗H)"""
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
    """6-CNOT Toffoli decomposition."""
    gates = []
    gates.append(_h(t))
    gates.append(_cnot(c1, t))
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(_cnot(c2, t))
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, t))
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(_cnot(c1, c2))
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, c2))
    return gates


def _match_known(target, qubits):
    """Try to match the target against known gate decompositions."""
    if qubits == 2:
        candidates = [
            (_decomp_cnot_01(), 1),
            (_decomp_cnot_10(), 1),
            (_decomp_swap(), 3),
            (_decomp_iswap(), 2),
        ]
        for gates, cnot_count in candidates:
            U = _build(gates, qubits)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
        for c1 in range(3):
            for c2 in range(3):
                if c2 == c1:
                    continue
                t = 3 - c1 - c2
                gates = _decomp_toffoli(c1, c2, t)
                U = _build(gates, qubits)
                if _distance(target, U) < 1e-10:
                    return gates
    return None


def _build_from_params(params, layout, qubits):
    """Build unitary from parameter values and a fixed layout."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return _build(gates, qubits)


def _generic_synthesis(target, qubits, budget):
    """Improved coordinate descent synthesis with better initialization and multi-start."""
    n = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout: interleaved u3 gates and CNOTs
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    best_dist = 1.0
    best_params = [0.0] * num_params
    rng = random.Random(42)
    
    # Multi-start: try several random initializations
    num_starts = 3
    time_per_start = 60.0 / num_starts
    
    for start in range(num_starts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Initialize parameters
        if start == 0:
            params = [0.0] * num_params
        else:
            params = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
        
        U = _build_from_params(params, layout, qubits)
        dist = _distance(target, U)
        
        if dist < best_dist:
            best_dist = dist
            best_params = params[:]
        
        # Coordinate descent
        max_sweeps = 20
        start_time = time.time()
        for sweep in range(max_sweeps):
            if time.time() - start_time > time_per_start:
                break
            if time.time() > _DEADLINE - 10:
                break
            
            step = 1.0 / (1.0 + sweep * 0.5)
            improved = False
            
            for i in range(num_params):
                if time.time() - start_time > time_per_start:
                    break
                base = params[i]
                trials = [base + step, base - step, base + step * 0.5, base - step * 0.5]
                if sweep < 3:
                    trials.append(base + rng.uniform(-math.pi, math.pi))
                
                for trial in trials:
                    params[i] = trial
                    U = _build_from_params(params, layout, qubits)
                    d = _distance(target, U)
                    if d < dist:
                        dist = d
                        if d < best_dist:
                            best_dist = d
                            best_params = params[:]
                        improved = True
                    else:
                        params[i] = base
            
            if not improved and sweep > 3:
                break
        
        # Polish
        for polish in range(5):
            if time.time() - start_time > time_per_start:
                break
            step = 0.05 / (1.0 + polish)
            for i in range(num_params):
                if time.time() - start_time > time_per_start:
                    break
                base = params[i]
                for trial in (base + step, base - step):
                    params[i] = trial
                    U = _build_from_params(params, layout, qubits)
                    d = _distance(target, U)
                    if d < dist:
                        dist = d
                        if d < best_dist:
                            best_dist = d
                            best_params = params[:]
                    else:
                        params[i] = base
    
    # Build final gate list from best_params
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_params[k], best_params[k + 1], best_params[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try known gate matching first
    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    return _generic_synthesis(target, qubits, budget)
