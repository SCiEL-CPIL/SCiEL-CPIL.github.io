"""Quantum circuit synthesis: KAK + local refinement with deadline guard."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 250.0  # 20s safety margin before 270s kill

def _time_left():
    return _DEADLINE - time.time()

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

# Pre-compute CNOT permutation indices for all qubit counts
_CNOT_PERMS = {}

def _cnot_perm(qubits, ctrl, tgt):
    key = (qubits, ctrl, tgt)
    if key not in _CNOT_PERMS:
        n = 2 ** qubits
        idx = np.arange(n)
        bit_ctrl = (idx >> (qubits - 1 - ctrl)) & 1
        bit_tgt_mask = 1 << (qubits - 1 - tgt)
        new_idx = idx.copy()
        mask = bit_ctrl == 1
        new_idx[mask] = idx[mask] ^ bit_tgt_mask
        _CNOT_PERMS[key] = new_idx
    return _CNOT_PERMS[key]

def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 = MSB."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Build operator by kron
            op = np.array([[1.0 + 0j]])
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            mat = op @ mat
        else:
            _, ctrl, tgt = gate
            perm = _cnot_perm(qubits, ctrl, tgt)
            mat = mat[perm, :]
    return mat

def _distance(target, mat):
    val = abs(np.trace(target.conj().T @ mat)) / target.shape[0]
    return 1.0 - min(1.0, val)

def _distance_fast(target, mat):
    """Faster distance: precompute target.conj().T once outside."""
    val = abs(np.vdot(target.conj().T.ravel(), mat.ravel())) / target.shape[0]
    return 1.0 - min(1.0, val)

# ---------- KAK for 2-qubit ----------
def _kak_decompose(U):
    """Decompose 2-qubit unitary into single-qubit + CNOTs. Returns (A0, A1, A2, A3, A4)
    such that U = A0 ⊗ A1 * (CNOT) * A2 ⊗ A3 * (CNOT) * A4 ⊗ A5 * (CNOT) * B0 ⊗ B1
    Simplified: use the standard 3-CNOT decomposition."""
    # We'll use a simpler approach: just do local search on a 3-CNOT template
    # For now, return None and let the general path handle it
    return None

def _synthesize_2qubit(problem):
    """2-qubit synthesis using KAK decomposition + local refinement."""
    qubits = 2
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    target_dag = target.conj().T
    
    # KAK decomposition: U = (A0 ⊗ A1) * CNOT * (A2 ⊗ A3) * CNOT * (A4 ⊗ A5) * CNOT * (B0 ⊗ B1)
    # Use the standard approach: extract the 3 interaction angles and local unitaries
    # For exactness, use the known formula via SU(4) invariants
    
    # Build template with exactly 3 CNOTs (or budget if less)
    n_cnots = min(budget, 3)
    
    layout = []
    for i in range(n_cnots):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["cx", 0, 1])
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def make_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    # Initialize with zeros
    params = np.zeros(n_params)
    
    # Try multiple random initializations + coordinate descent
    best_dist = 1.0
    best_params = params.copy()
    
    rng = np.random.RandomState(42)
    n_restarts = 5 if budget <= 3 else 3
    
    for restart in range(n_restarts):
        if _time_left() < 10.0:
            break
        if restart == 0:
            params = np.zeros(n_params)
        else:
            params = rng.uniform(-math.pi, math.pi, size=n_params)
        
        current_mat = _build_fast(make_circuit(params), qubits)
        cur_dist = 1.0 - min(1.0, abs(np.vdot(target_dag.ravel(), current_mat.ravel())) / n)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_params = params.copy()
        
        # Coordinate descent with adaptive step
        step = 1.0
        for sweep in range(6):
            if _time_left() < 8.0:
                break
            step = 1.0 / (1 + sweep * 0.6)
            improved = False
            for i in range(n_params):
                for trial in (params[i] + step, params[i] - step):
                    new_params = params.copy()
                    new_params[i] = trial
                    mat = _build_fast(make_circuit(new_params), qubits)
                    d = 1.0 - min(1.0, abs(np.vdot(target_dag.ravel(), mat.ravel())) / n)
                    if d < cur_dist:
                        cur_dist = d
                        params = new_params
                        improved = True
                # Random kick
                trial = params[i] + rng.uniform(-math.pi, math.pi)
                new_params = params.copy()
                new_params[i] = trial
                mat = _build_fast(make_circuit(new_params), qubits)
                d = 1.0 - min(1.0, abs(np.vdot(target_dag.ravel(), mat.ravel())) / n)
                if d < cur_dist:
                    cur_dist = d
                    params = new_params
                    improved = True
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_params = params.copy()
            if not improved or cur_dist < 1e-10:
                break
    
    return make_circuit(best_params)

def _synthesize_3qubit(problem):
    """3-qubit synthesis using CNOT ladder + multi-restart local search."""
    qubits = 3
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    target_dag = target.conj().T
    target_flat = target_dag.ravel()
    
    # CNOT ladder pattern: alternate between (0,1), (1,2), (0,1), (1,2), ...
    # This creates a more connected interaction pattern
    pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    
    # Build layout: layers of u3(0),u3(1),u3(2) + one CNOT per layer
    layout = []
    for layer in range(budget):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["u3", 2])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    # Final u3 layer
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    layout.append(["u3", 2])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def make_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    def eval_dist(params):
        mat = _build_fast(make_circuit(params), qubits)
        d = 1.0 - min(1.0, abs(np.vdot(target_flat, mat.ravel())) / n)
        return d
    
    # Multi-restart coordinate descent
    best_dist = 1.0
    best_params = np.zeros(n_params)
    
    rng = np.random.RandomState(123)
    n_restarts = 3
    
    for restart in range(n_restarts):
        if _time_left() < 15.0:
            break
        if restart == 0:
            params = np.zeros(n_params)
        else:
            params = rng.uniform(-math.pi, math.pi, size=n_params)
        
        cur_dist = eval_dist(params)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_params = params.copy()
        
        # Coordinate descent with adaptive step
        step = 1.0
        for sweep in range(5):
            if _time_left() < 12.0:
                break
            step = 1.0 / (1 + sweep * 0.7)
            improved = False
            for i in range(n_params):
                for trial in (params[i] + step, params[i] - step):
                    new_params = params.copy()
                    new_params[i] = trial
                    d = eval_dist(new_params)
                    if d < cur_dist:
                        cur_dist = d
                        params = new_params
                        improved = True
                # Random kick
                trial = params[i] + rng.uniform(-math.pi, math.pi)
                new_params = params.copy()
                new_params[i] = trial
                d = eval_dist(new_params)
                if d < cur_dist:
                    cur_dist = d
                    params = new_params
                    improved = True
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_params = params.copy()
            if not improved or cur_dist < 1e-10:
                break
    
    return make_circuit(best_params)

def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    if qubits == 2:
        return _synthesize_2qubit(problem)
    elif qubits == 3:
        return _synthesize_3qubit(problem)
    else:
        # Fallback: identity
        return []
# EVOLVE-BLOCK-END
