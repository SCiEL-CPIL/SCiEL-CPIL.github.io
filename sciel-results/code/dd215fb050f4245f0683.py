# EVOLVE-BLOCK-START
import math
import random
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_TIME_LIMIT = 230.0  # Keep 40s margin from 270s wall-clock kill


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    
    # Pre-allocate indices for CNOT permutations? 
    # For small N, direct construction is fine.
    # We can speed up U3 application by not building the full operator matrix via kron every time.
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            # U3 matrix
            # [[c, -e^{i lam} s], [e^{i phi} s, e^{i(phi+lam)} c]]
            
            # Apply to matrix columns
            # The qubit q is the (qubits - 1 - q)-th bit from the right (LSB) in the index?
            # Problem says: "qubit 0 is the most significant bit of the basis index"
            # So index = q0 * 2^(n-1) + q1 * 2^(n-2) + ...
            # Qubit 0 is MSB.
            
            # We can iterate over all basis states and update the matrix entries
            # But applying a single qubit gate to a matrix is:
            # U_new[i, j] = sum_k U_old[k, j] * G[i, k]
            # Since G is local, only 2 rows/cols mix.
            
            # Let's stick to the operator method but optimize it.
            # Create the operator
            op = np.eye(n, dtype=complex)
            # Identify the indices affected
            # Qubit q is at position q from MSB.
            # Bit mask for this qubit: 1 << (qubits - 1 - q)
            mask = 1 << (qubits - 1 - q)
            
            # Iterate over all indices
            # For each index, if the bit q is 0, it maps to itself (row 0 of U3)
            # If bit q is 1, it maps to itself (row 1 of U3)
            # But we need to mix the two states where only qubit q differs.
            
            # Faster: Iterate over all indices i
            # For each i, we need to update matrix[i, :]
            # The gate acts on the i-th row (output) from the input rows.
            # Actually, matrix is U. U = G @ U_prev.
            # So U_new[i, j] = sum_k G[i, k] U_prev[k, j].
            
            # G is identity except for the 2x2 block at qubit q.
            # Let's just build the 2x2 block and apply it.
            
            # Indices where qubit q is 0: i0
            # Indices where qubit q is 1: i1 = i0 | mask
            
            # We need to update all rows i.
            # If i has bit q = 0:
            #   U_new[i, j] = c * U_prev[i, j] - e^{i lam} s * U_prev[i|mask, j]
            # If i has bit q = 1:
            #   U_new[i, j] = e^{i phi} s * U_prev[i^mask, j] + e^{i(phi+lam)} c * U_prev[i, j]
            
            # This is O(N^2) per gate, but with low constant.
            # Let's implement this vectorized.
            
            # Create masks for rows
            # rows_0: indices where bit q is 0
            # rows_1: indices where bit q is 1
            
            # We can just loop over the N rows. N is 4 or 8.
            for i in range(n):
                if (i & mask) == 0:
                    i_partner = i | mask
                    # U[i, :] = c * U[i, :] - exp(1j*lam)*s * U[i_partner, :]
                    matrix[i, :] = c * matrix[i, :] - np.exp(1j * lam) * s * matrix[i_partner, :]
                else:
                    i_partner = i ^ mask
                    # U[i, :] = exp(1j*phi)*s * U[i_partner, :] + exp(1j*(phi+lam))*c * U[i, :]
                    matrix[i, :] = np.exp(1j * phi) * s * matrix[i_partner, :] + np.exp(1j * (phi + lam)) * c * matrix[i, :]
                    
        else:
            _, control, target = gate
            # CNOT
            # Control bit c, Target bit t
            # If control is 1, flip target.
            # This is a permutation of rows.
            # Row i maps to Row perm[i]
            # U_new[i, j] = U_prev[perm[i], j]
            
            mask_c = 1 << (qubits - 1 - control)
            mask_t = 1 << (qubits - 1 - target)
            
            # We need to swap rows where control=1 and target=0/1?
            # No, CNOT: |c, t> -> |c, t^c>
            # So if c=1, t flips.
            # Pairs of rows to swap:
            # For all other bits fixed, if c=1:
            #   Row (..., 1, 0, ...) swaps with Row (..., 1, 1, ...)
            
            # Let's generate the permutation
            # Since N is small, we can just build the perm array
            perm = np.arange(n)
            for idx in range(n):
                if (idx & mask_c):
                    perm[idx] = idx ^ mask_t
            
            # Apply permutation to rows
            # matrix = matrix[perm, :]
            # This creates a new array.
            # We can do in-place if we are careful, or just reassign.
            matrix = matrix[perm, :]

    return matrix

def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / 2^n"""
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _cnot_gate(control, target):
    return ["cx", control, target]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _exact_cnot():
    """Exact CNOT decomposition: 1 CNOT needed."""
    return [_cnot_gate(0, 1)], 1


def _exact_iswap():
    """iSWAP decomposition: 3 CNOTs + single-qubit rotations."""
    gates = []
    gates.append(_u3_gate(0, math.pi, 0, 0))
    gates.append(_u3_gate(1, math.pi, 0, 0))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(0, 0, 0, -math.pi / 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(1, 0))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(0, math.pi, 0, 0))
    gates.append(_u3_gate(1, math.pi, 0, 0))
    return gates, 3


def _exact_swap():
    """SWAP = 3 CNOTs, no single-qubit rotations needed."""
    gates = [
        _cnot_gate(0, 1),
        _cnot_gate(1, 0),
        _cnot_gate(0, 1),
    ]
    return gates, 3


def _exact_toffoli():
    """Toffoli (CCNOT) decomposition: 6 CNOTs."""
    gates = []
    gates.append(_u3_gate(2, math.pi, 0, 0))
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(0, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, -math.pi / 4))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(0, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_u3_gate(2, math.pi, 0, 0))
    return gates, 6


def _is_known_gate(problem):
    """Check if the target is one of the known gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    target = _target(problem)

    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"

        iswap = np.array([[1, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"

        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"

    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"

    return None


def _optimize_layout(target, qubits, layout_gates, rng):
    """
    Optimize the parameters of a fixed layout using scipy.
    layout_gates: list of gates, with u3 gates having placeholder parameters.
    """
    u3_indices = []
    param_idx = 0
    for i, gate in enumerate(layout_gates):
        if gate[0] == "u3":
            u3_indices.append((i, param_idx))
            param_idx += 3

    n_params = param_idx
    if n_params == 0:
        return layout_gates, _distance(target, _build(layout_gates, qubits))

    # Objective function
    def objective(params):
        gates = [list(g) for g in layout_gates]
        k = 0
        for i, gate in enumerate(gates):
            if gate[0] == "u3":
                gates[i] = [gate[0], gate[1], params[k], params[k + 1], params[k + 2]]
                k += 3
        dist = _distance(target, _build(gates, qubits))
        return dist

    # Initial guess
    x0 = np.zeros(n_params)
    
    # Bounds for u3 parameters: theta, phi, lam in [-pi, pi]
    bounds = [(-math.pi, math.pi)] * n_params

    # Try differential evolution for global search if time permits, else minimize
    # For small n_params, minimize is often enough with good initial guess.
    # Let's do a few random restarts with minimize.
    
    best_dist = 1.0
    best_params = None
    
    max_restarts = 10
    for r in range(max_restarts):
        if _remaining_time() < 10:
            break
            
        if r == 0:
            x_start = x0
        else:
            x_start = rng.uniform(-math.pi, math.pi, n_params)
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and res.fun < best_dist:
                best_dist = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = x0
        best_dist = objective(x0)

    # Construct final gates
    final_gates = [list(g) for g in layout_gates]
    k = 0
    for i, gate in enumerate(final_gates):
        if gate[0] == "u3":
            final_gates[i] = [gate[0], gate[1], best_params[k], best_params[k + 1], best_params[k + 2]]
            k += 3
            
    return final_gates, best_dist


def _generate_layouts(qubits, budget, rng):
    """Generate diverse layouts for optimization."""
    layouts = []
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout 1: Standard alternating
    gates = []
    for q in range(qubits):
        gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
    for i in range(budget):
        for q in range(qubits):
            gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
        c, t = pairs[rng.randint(0, len(pairs) - 1)]
        gates.append(_cnot_gate(c, t))
    for q in range(qubits):
        gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
    layouts.append(gates)
    
    # Layout 2: More structured (e.g. for 2 qubits, 3 CNOTs)
    if qubits == 2 and budget >= 3:
        gates = []
        for q in range(2): gates.append(_u3_gate(q, 0,0,0))
        gates.append(_cnot_gate(0, 1))
        for q in range(2): gates.append(_u3_gate(q, 0,0,0))
        gates.append(_cnot_gate(1, 0))
        for q in range(2): gates.append(_u3_gate(q, 0,0,0))
        gates.append(_cnot_gate(0, 1))
        for q in range(2): gates.append(_u3_gate(q, 0,0,0))
        layouts.append(gates)
        
    return layouts


def _synthesize_generic(target, qubits, budget, rng):
    """Synthesize a generic unitary using multiple random restarts and greedy optimization."""
    best_gates = None
    best_dist = 1.0

    layouts = _generate_layouts(qubits, budget, rng)
    
    # Add some random layouts
    for _ in range(3):
        gates = []
        pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
        for q in range(qubits):
            gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
        for i in range(budget):
            for q in range(qubits):
                gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
            c, t = pairs[rng.randint(0, len(pairs) - 1)]
            gates.append(_cnot_gate(c, t))
        for q in range(qubits):
            gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
        layouts.append(gates)

    for layout in layouts:
        if _remaining_time() < 15:
            break
        gates, dist = _optimize_layout(target, qubits, layout, rng)
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
        if best_dist < 1e-10:
            break

    if best_gates is None:
        best_gates = [_u3_gate(q, 0.0, 0.0, 0.0) for q in range(qubits)]

    return best_gates


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Clamp budget to reasonable values
    if budget < 0:
        budget = 0
    if budget > 100:
        budget = 100

    rng = random.Random(12345 + qubits * 100 + budget)

    known = _is_known_gate(problem)
    if known == "cnot":
        gates, cnot_count = _exact_cnot()
        if cnot_count <= budget:
            return gates
    elif known == "iswap":
        gates, cnot_count = _exact_iswap()
        if cnot_count <= budget:
            return gates
    elif known == "swap":
        gates, cnot_count = _exact_swap()
        if cnot_count <= budget:
            return gates
    elif known == "toffoli":
        gates, cnot_count = _exact_toffoli()
        if cnot_count <= budget:
            return gates

    gates = _synthesize_generic(target, qubits, budget, rng)

    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        filtered = []
        for g in gates:
            if g[0] == "cx":
                if count < budget:
                    filtered.append(g)
                    count += 1
            else:
                filtered.append(g)
        gates = filtered

    if len(gates) > 400:
        gates = gates[:400]
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count > budget:
            count = 0
            filtered = []
            for g in gates:
                if g[0] == "cx":
                    if count < budget:
                        filtered.append(g)
                        count += 1
                else:
                    filtered.append(g)
            gates = filtered

    return gates
# EVOLVE-BLOCK-END
