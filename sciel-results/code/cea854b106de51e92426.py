"""Quantum circuit synthesis using exact known circuits and KAK/optimization fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 240.0  # Safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            eye = np.eye(2, dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            idx = np.arange(n)
            ctrl_bit = (idx >> (qubits - 1 - control)) & 1
            tgt_shift = (qubits - 1 - target)
            mask = (ctrl_bit == 1)
            new_idx = idx.copy()
            new_idx[mask] = idx[mask] ^ (1 << tgt_shift)
            U = U[new_idx, :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(val) / dim)


def _check_close(A, B, tol=1e-5):
    """Check if two unitaries are equal up to global phase."""
    dim = A.shape[0]
    max_idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[max_idx]) < 1e-12:
        return False
    phase = A[max_idx] / B[max_idx]
    if abs(phase) < 1e-12:
        return False
    phase = phase / abs(phase)
    return np.allclose(A, phase * B, atol=tol)


def _known_circuit_cnot_2q():
    return [["cx", 0, 1]]


def _known_circuit_swap_2q():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_iswap_2q():
    pi = math.pi
    return [
        ["u3", 0, pi / 2, 0, pi / 2],
        ["u3", 1, pi / 2, 0, -pi / 2],
        ["cx", 0, 1],
        ["u3", 0, pi / 2, 0, -pi / 2],
        ["u3", 1, pi / 2, 0, pi / 2],
        ["cx", 1, 0],
    ]


def _known_circuit_toffoli_3q():
    pi = math.pi
    return [
        ["u3", 2, pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["u3", 2, pi / 2, 0, 0],
    ]


def _fredkin_circuit(ctrl, a, b):
    """Fredkin (controlled-SWAP) circuit."""
    return [
        ["cx", a, b],
        ["cx", ctrl, b],
        ["cx", a, b],
        ["cx", ctrl, a],
        ["cx", a, b],
        ["cx", ctrl, b],
    ]


def _ccz_circuit(c1, c2, t):
    pi = math.pi
    return [
        ["u3", t, 0, 0, pi],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
        ["u3", t, 0, 0, pi / 2],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
    ]


def _detect_known_gate(target, qubits, budget):
    """Try to identify the target as a known gate and return exact circuit if so."""
    if qubits == 2:
        if budget >= 1:
            ref = _build_fast([["cx", 0, 1]], 2)
            if _check_close(target, ref):
                return _known_circuit_cnot_2q()
            ref = _build_fast([["cx", 1, 0]], 2)
            if _check_close(target, ref):
                return [["cx", 1, 0]]
        if budget >= 3:
            circ = _known_circuit_swap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
        if budget >= 2:
            circ = _known_circuit_iswap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
    elif qubits == 3:
        if budget >= 6:
            # Toffoli (CCX) with q0,q1 as controls, q2 as target
            circ = _known_circuit_toffoli_3q()
            ref = _build_fast(circ, 3)
            if _check_close(target, ref):
                return circ
            # Other control/target combinations for Toffoli
            for c1, c2, tgt in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                circ_alt = _toffoli_circuit(c1, c2, tgt)
                ref = _build_fast(circ_alt, 3)
                if _check_close(target, ref):
                    return circ_alt
            # Fredkin variants
            for ctrl, a, b in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                circ = _fredkin_circuit(ctrl, a, b)
                ref = _build_fast(circ, 3)
                if _check_close(target, ref):
                    return circ
            # CCZ variants
            for c1, c2, t in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                circ = _ccz_circuit(c1, c2, t)
                ref = _build_fast(circ, 3)
                if _check_close(target, ref):
                    return circ
    return None


def _toffoli_circuit(c1, c2, tgt):
    """Toffoli circuit for arbitrary control/target qubits."""
    pi = math.pi
    # Standard decomposition adapted for arbitrary qubit assignment
    # We use the same structure as the standard circuit but with permuted qubits
    # For (c1, c2, tgt), we map standard (0,1,2) to (c1,c2,tgt)
    # The standard circuit uses: u3 on tgt, cx(c2,tgt), u3 on tgt, cx(c1,tgt), u3 on tgt, cx(c2,tgt), u3 on tgt, cx(c1,tgt), u3 on tgt, u3 on tgt
    return [
        ["u3", tgt, pi / 2, 0, 0],
        ["cx", c2, tgt],
        ["u3", tgt, pi / 4, 0, 0],
        ["cx", c1, tgt],
        ["u3", tgt, -pi / 4, 0, 0],
        ["cx", c2, tgt],
        ["u3", tgt, pi / 4, 0, 0],
        ["cx", c1, tgt],
        ["u3", tgt, -pi / 4, 0, 0],
        ["u3", tgt, pi / 2, 0, 0],
    ]


def _u3_from_matrix(mat):
    """Extract (theta, phi, lam) from a 2x2 unitary matrix."""
    # U = [[cos(theta/2), -e^{i*lam} sin(theta/2)],
    #      [e^{i*phi} sin(theta/2), e^{i*(phi+lam)} cos(theta/2)]]
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    
    # theta from |a| = cos(theta/2)
    cos_half = abs(a)
    if cos_half > 1.0:
        cos_half = 1.0
    if cos_half < 0.0:
        cos_half = 0.0
    theta = 2 * math.acos(cos_half)
    
    if abs(math.sin(theta / 2)) < 1e-10:
        # theta ~ 0 or 2pi, degenerate case
        if abs(a) > 0.5:
            # theta ~ 0
            phi = 0.0
            lam = 0.0
        else:
            # theta ~ 2pi
            phi = 0.0
            lam = 0.0
        return theta, phi, lam
    
    sin_half = math.sin(theta / 2)
    
    # From a = cos(theta/2), phase of a gives relative info
    # From b = -e^{i*lam} sin(theta/2)
    if abs(b) > 1e-10:
        lam = (math.angle(-b / sin_half) + math.pi) % (2 * math.pi) - math.pi
        if lam < 0:
            lam += 2 * math.pi
    else:
        lam = 0.0
    
    # From c = e^{i*phi} sin(theta/2)
    if abs(c) > 1e-10:
        phi = math.angle(c / sin_half)
    else:
        phi = 0.0
    
    return theta, phi, lam


def _kak_2qubit(target, qubits=2):
    """KAK decomposition for 2-qubit unitary. Returns list of gates."""
    # For 2 qubits, we can use the standard decomposition
    # U = e^{i*alpha} * (A ⊗ B) * exp(-i*sum_k t_k * (sigma_k ⊗ sigma_k)) * (C ⊗ D)
    # where t_k are the KAK angles
    
    # Simplified approach: use known decomposition structure
    # For 2 qubits with up to 3 CNOTs, we can use:
    # U = (A0 ⊗ A1) * CX(0,1) * (B0 ⊗ B1) * CX(0,1) * (C0 ⊗ C1) * CX(0,1) * (D0 ⊗ D1)
    
    # This is complex to solve exactly. Let's use a numerical approach.
    # For now, return a template that optimization can work with.
    # Actually, let's try a different approach: direct parameter optimization
    
    return None


def _optimize_params(gates_template, target, qubits, max_iter=50):
    """Optimize u3 parameters using gradient-free method."""
    n_params = 3 * sum(1 for g in gates_template if g[0] == "u3")
    
    def circuit_from_params(params):
        gates, k = [], 0
        for gate in gates_template:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with zeros
    best_params = [0.0] * n_params
    best_dist = _distance(target, _build_fast(circuit_from_params(best_params), qubits))
    
    # Coordinate descent with adaptive step size
    step = 1.0
    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        improved = False
        for i in range(n_params):
            if _time_left() < 3:
                break
            for direction in (1, -1):
                trial = best_params[:]
                trial[i] += direction * step
                dist = _distance(target, _build_fast(circuit_from_params(trial), qubits))
                if dist < best_dist:
                    best_params = trial
                    best_dist = dist
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return circuit_from_params(best_params), best_dist


def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Fallback: build a template and optimize
    # For 2 qubits: use up to 3 CNOTs
    # For 3 qubits: use up to budget CNOTs
    
    pairs_2q = [(0, 1), (1, 0)]
    pairs_3q = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    pairs = pairs_2q if qubits == 2 else pairs_3q
    
    # Build template: alternate u3 layers and CNOTs
    template = []
    cnot_used = 0
    
    # Initial u3 layer
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Add CNOTs up to budget
    for i in range(budget):
        if cnot_used >= budget:
            break
        c, t = pairs[i % len(pairs)]
        template.append(["cx", c, t])
        cnot_used += 1
        # Add u3 layer after each CNOT
        template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Final u3 layer
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Check gate count
    if len(template) > 400:
        # Truncate if too many gates
        template = template[:400]
    
    # Optimize
    result, dist = _optimize_params(template, target, qubits, max_iter=80)
    
    return result
# EVOLVE-BLOCK-END
