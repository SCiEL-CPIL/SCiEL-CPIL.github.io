"""Rewritten program for exact quantum circuit synthesis."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 210.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]

def _cx(c, t):
    return ["cx", int(c), int(t)]

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _check_global_phase_equivalence(target, candidate):
    """Check if candidate is equivalent to target up to a global phase."""
    dim = target.shape[0]
    trace_val = np.trace(target.conj().T @ candidate)
    if abs(trace_val) < 1e-10:
        return False
    phase = trace_val / abs(trace_val)
    # Check if candidate * phase.conj() is close to target
    diff = np.linalg.norm(target - candidate * phase.conj())
    return diff < 1e-8

def _synthesize_cnot_2q(target):
    """Exact synthesis for 2-qubit CNOT (0->1)."""
    return [_cx(0, 1)]

def _synthesize_swap_2q(target):
    """Exact synthesis for 2-qubit SWAP (0,1)."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

def _synthesize_iswap_2q(target):
    """Exact synthesis for 2-qubit iSWAP (0,1)."""
    # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    # Decomposition using 2 CNOTs
    gates = []
    # H on qubit 1
    gates.append(_u3(1, math.pi, 0, 0)) # H is u3(pi, 0, pi) or similar? H = 1/sqrt(2)[[1,1],[1,-1]]. u3(pi, 0, pi) = [[0, -1], [1, 0]]? No.
    # Standard H: u3(pi, 0, pi) is NOT H.
    # H = exp(i pi/2 (X+Z)/sqrt(2))...
    # u3(theta, phi, lam):
    # [[cos(theta/2), -exp(i*lam)sin(theta/2)], [exp(i*phi)sin(theta/2), exp(i*(phi+lam))cos(theta/2)]]
    # For H: theta=pi, phi=0, lam=pi/2?
    # c=0, s=1. [[0, -exp(i*pi/2)], [1, 0]] = [[0, -i], [1, 0]]. This is not H.
    # H = 1/sqrt(2) [[1,1],[1,-1]].
    # Let's use a known decomposition.
    # iSWAP = (I \otimes H) CNOT (I \otimes S) CNOT (I \otimes H) ... no.
    # Known 2-CNOT iSWAP:
    # 1. u3(pi/2, 0, pi/2) on q0
    # 2. u3(pi/2, pi, pi/2) on q1
    # 3. cx(0,1)
    # 4. u3(pi/2, pi, 0) on q0
    # 5. u3(pi/2, 0, pi) on q1
    # 6. cx(0,1)
    # Let's verify this.
    gates = [
        _u3(0, math.pi/2, 0, math.pi/2),
        _u3(1, math.pi/2, math.pi, math.pi/2),
        _cx(0, 1),
        _u3(0, math.pi/2, math.pi, 0),
        _u3(1, math.pi/2, 0, math.pi),
        _cx(0, 1)
    ]
    return gates

def _synthesize_toffoli_3q(target):
    """Exact synthesis for 3-qubit Toffoli (0,1->2)."""
    # Standard 6-CNOT Toffoli
    gates = []
    # 1. H on 2
    gates.append(_u3(2, math.pi, 0, 0)) # Approx H? No, let's use exact H parameters.
    # H = u3(pi, 0, pi/2) is wrong.
    # H matrix: 1/sqrt(2) [[1,1],[1,-1]]
    # u3(pi, 0, pi) = [[0, -1], [1, 0]]
    # u3(pi, 0, pi/2) = [[0, -i], [1, 0]]
    # Let's find u3 for H.
    # c=1/sqrt(2), s=1/sqrt(2).
    # -exp(i*lam)*s = 1/sqrt(2) => exp(i*lam) = -1 => lam = pi.
    # exp(i*phi)*s = 1/sqrt(2) => exp(i*phi) = 1 => phi = 0.
    # exp(i*(phi+lam))*c = -1/sqrt(2) => exp(i*pi)*1/sqrt(2) = -1/sqrt(2). Correct.
    # So H = u3(pi, 0, pi).
    
    # Toffoli decomposition (0,1 -> 2):
    # 1. H(2)
    # 2. C(1,2)
    # 3. T^dag(2)
    # 4. C(0,2)
    # 5. T(2)
    # 6. C(1,2)
    # 7. T^dag(2)
    # 8. T(1)
    # 9. C(0,1)
    # 10. T^dag(1)
    # 11. T(2)
    # 12. C(0,2)
    # 13. H(2)
    
    # T = diag(1, exp(i*pi/4))
    # T^dag = diag(1, exp(-i*pi/4))
    # u3 for T: theta=0, phi=0, lam=pi/2?
    # u3(0, 0, pi/2) = [[1, 0], [0, exp(i*pi/2)]] = [[1,0],[0,i]]. No.
    # T = [[1,0],[0, e^{i pi/4}]].
    # u3(0, 0, pi/4) = [[1,0],[0, e^{i pi/4}]]. Yes.
    # So T = u3(0, 0, pi/4).
    # T^dag = u3(0, 0, -pi/4).
    
    H = _u3(2, math.pi, 0, math.pi)
    T = _u3(2, 0, 0, math.pi/4)
    Tdag = _u3(2, 0, 0, -math.pi/4)
    
    gates = [
        H,
        _cx(1, 2),
        Tdag,
        _cx(0, 2),
        T,
        _cx(1, 2),
        Tdag,
        _u3(1, 0, 0, math.pi/4), # T on 1
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4), # Tdag on 1
        T,
        _cx(0, 2),
        H
    ]
    return gates

def _optimize_generic_2q(target, qubits, budget):
    """Optimize 2-qubit unitary with 1 CNOT."""
    if _elapsed() > _DEADLINE:
        return []
    
    layout = [
        ["u3", 0, 0,0,0], ["u3", 1, 0,0,0],
        ["cx", 0, 1],
        ["u3", 0, 0,0,0], ["u3", 1, 0,0,0]
    ]
    
    def build_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(g)
        return gates
    
    def objective(params):
        gates = build_circuit(params)
        u = _build(gates, qubits)
        return _distance(target, u)
    
    best_params = np.zeros(6)
    best_dist = objective(best_params)
    
    # Multi-start
    for i in range(5):
        if _elapsed() > _DEADLINE:
            break
        init = np.random.uniform(-math.pi, math.pi, 6)
        res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-8})
        if res.fun < best_dist:
            best_dist = res.fun
            best_params = res.x
            
    return build_circuit(best_params)

def _optimize_generic_3q(target, qubits, budget):
    """Optimize 3-qubit unitary with 6 CNOTs."""
    if _elapsed() > _DEADLINE:
        return []
    
    # Layout: 6 CNOTs
    # C(0,1), C(1,2), C(0,2), C(0,1), C(1,2), C(0,2)
    layout = []
    cnots = [(0,1), (1,2), (0,2), (0,1), (1,2), (0,2)]
    
    for i in range(len(cnots) + 1):
        for q in range(qubits):
            layout.append(["u3", q, 0,0,0])
        if i < len(cnots):
            layout.append(["cx", cnots[i][0], cnots[i][1]])
            
    num_params = 3 * qubits * (len(cnots) + 1)
    
    def build_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(g)
        return gates
    
    def objective(params):
        gates = build_circuit(params)
        u = _build(gates, qubits)
        return _distance(target, u)
    
    best_params = np.zeros(num_params)
    best_dist = objective(best_params)
    
    # Multi-start with few iterations
    for i in range(3):
        if _elapsed() > _DEADLINE:
            break
        init = np.random.uniform(-math.pi, math.pi, num_params)
        res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-6})
        if res.fun < best_dist:
            best_dist = res.fun
            best_params = res.x
            
    return build_circuit(best_params)

def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check known gates
    if qubits == 2:
        # CNOT
        cnot_target = np.eye(4, dtype=complex)
        cnot_target[2,3] = 1
        cnot_target[3,2] = 1
        if _check_global_phase_equivalence(target, cnot_target):
            return _synthesize_cnot_2q(target)
            
        # SWAP
        swap_target = np.eye(4, dtype=complex)
        swap_target[1,2] = 1
        swap_target[2,1] = 1
        swap_target[1,1] = 0
        swap_target[2,2] = 0
        if _check_global_phase_equivalence(target, swap_target):
            return _synthesize_swap_2q(target)
            
        # iSWAP
        iswap_target = np.zeros(4, dtype=complex)
        iswap_target[1,2] = 1j
        iswap_target[2,1] = 1j
        if _check_global_phase_equivalence(target, iswap_target):
            return _synthesize_iswap_2q(target)
            
        # Generic 2-qubit
        return _optimize_generic_2q(target, qubits, budget)
        
    elif qubits == 3:
        # Toffoli (0,1 -> 2)
        toffoli_target = np.eye(8, dtype=complex)
        toffoli_target[6,7] = 1
        toffoli_target[7,6] = 1
        toffoli_target[6,6] = 0
        toffoli_target[7,7] = 0
        if _check_global_phase_equivalence(target, toffoli_target):
            return _synthesize_toffoli_3q(target)
            
        # Generic 3-qubit
        return _optimize_generic_3q(target, qubits, budget)
        
    return []
# EVOLVE-BLOCK-END
