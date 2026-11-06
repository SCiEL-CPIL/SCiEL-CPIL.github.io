"""Quantum circuit synthesis with exact decompositions and gradient-based optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 260.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _is_cnot_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_iswap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_swap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_toffoli_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 3:
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        return np.allclose(target, expected, atol=1e-10)
    return False


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """iSWAP decomposition using 2 CNOTs."""
    gates = []
    # H on both
    gates.append(["u3", 0, math.pi, 0, math.pi])
    gates.append(["u3", 1, math.pi, 0, math.pi])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # Phase: S on q0, S-dagger on q1
    gates.append(["u3", 0, 0, 0, math.pi / 2])
    gates.append(["u3", 1, 0, 0, -math.pi / 2])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # H on both
    gates.append(["u3", 0, math.pi, 0, math.pi])
    gates.append(["u3", 1, math.pi, 0, math.pi])
    return gates


def _decompose_swap_2qubit():
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _decompose_toffoli_3qubit():
    """Toffoli decomposition using 6 CNOTs."""
    gates = []
    # H on target (qubit 2)
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T-dagger on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T-dagger on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # H on target
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # T-dagger on qubit 1
    gates.append(["u3", 1, 0, 0, -math.pi / 4])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # T on qubit 1
    gates.append(["u3", 1, 0, 0, math.pi / 4])
    return gates


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(qubits, control, target):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _build_from_params(params, layout, qubits):
    """Build unitary from parameter vector and layout."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            k += 3
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for qq in range(qubits):
                operator = np.kron(operator, single if qq == gate[1] else eye)
            matrix = operator @ matrix
        else:
            matrix = _cnot_matrix(qubits, gate[1], gate[2]) @ matrix
    return matrix


def _fidelity(target, matrix):
    return abs(np.trace(target.conj().T @ matrix)) / target.shape[0]


def _gradient_descent(target, layout, qubits, params, time_limit):
    """Gradient-based optimization of rotation parameters."""
    dim = target.shape[0]
    n_params = len(params)
    best_params = params.copy()
    best_fid = _fidelity(target, _build_from_params(best_params, layout, qubits))
    
    # Compute gradient via finite differences (central difference)
    # But that's expensive. Let's use a smarter approach:
    # Actually, for 2-3 qubit systems, we can compute gradients analytically
    # by propagating the state through the circuit.
    
    # For efficiency, let's use a hybrid: Nelder-Mead style with adaptive steps
    # since true gradient computation for unitary fidelity is complex.
    
    # Let's use a more efficient optimization: L-BFGS-like with numerical gradients
    # But given time constraints, let's use a well-tuned Nelder-Mead
    
    return _nelder_mead(target, layout, qubits, best_params, best_fid, time_limit)


def _nelder_mead(target, layout, qubits, params, best_fid, time_limit):
    """Nelder-Mead optimization for the rotation parameters."""
    dim = target.shape[0]
    n = len(params)
    
    # Initialize simplex
    simplex = [params.copy()]
    for i in range(n):
        p = params.copy()
        p[i] += 0.5 if abs(p[i]) < 0.5 else 0.5 * p[i]
        simplex.append(p)
    
    # Evaluate initial simplex
    fvals = []
    for p in simplex:
        fvals.append(1.0 - _fidelity(target, _build_from_params(p, layout, qubits)))
    
    # Sort by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    alpha, gamma, rho = 1.0, 2.0, 0.5
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() - _start_time > time_limit:
            break
        
        # Reflection
        centroid = np.mean([np.array(simplex[i]) for i in range(n)], axis=0)
        worst = simplex[-1]
        xr = centroid + alpha * (centroid - np.array(worst))
        fr = 1.0 - _fidelity(target, _build_from_params(xr, layout, qubits))
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - np.array(worst))
            fe = 1.0 - _fidelity(target, _build_from_params(xe, layout, qubits))
            if fe < fr:
                simplex[-1] = list(xe)
                fvals[-1] = fe
            else:
                simplex[-1] = list(xr)
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = list(xr)
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (np.array(worst) - centroid)
            fc = 1.0 - _fidelity(target, _build_from_params(xc, layout, qubits))
            if fc < fvals[-1]:
                simplex[-1] = list(xc)
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = 1.0 - _fidelity(target, _build_from_params(simplex[i], layout, qubits))
        
        # Re-sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if max(fvals) - min(fvals) < 1e-10:
            break
    
    return simplex[0], fvals[0]


def _optimize_generic(target, qubits, budget, time_limit):
    """Optimize a generic unitary using Nelder-Mead on rotation parameters."""
    # Build layout: alternating single-qubit gates and CNOTs
    # For 2 qubits: use CNOT pairs (0,1) and (1,0)
    # For 3 qubits: use all 6 CNOT pairs
    
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else:
        pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    
    layout = []
    # Start with single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Add CNOT layers with single-qubit gates between
    for layer in range(budget):
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Final single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Count parameters
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initialize parameters
    rng = np.random.RandomState(42)
    params = rng.uniform(-math.pi, math.pi, size=n_params)
    
    # Optimize
    best_params, best_dist = _nelder_mead(target, layout, qubits, params, 0.0, time_limit)
    
    # Build gate list from best params
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
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check for known gates and use exact decompositions
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries, use Nelder-Mead optimization
    remaining = _DEADLINE - (time.time() - _start_time)
    if remaining < 5:
        # Not enough time, return a simple circuit
        gates = []
        for q in range(qubits):
            gates.append(["u3", q, 0.0, 0.0, 0.0])
        return gates

    return _optimize_generic(target, qubits, budget, _DEADLINE - 5)
# EVOLVE-BLOCK-END
