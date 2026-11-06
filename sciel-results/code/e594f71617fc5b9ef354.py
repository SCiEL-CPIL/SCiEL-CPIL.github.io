import math
import time
import random
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 250.0  # 20s margin

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(dim, dtype=complex)
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

def _is_fixed_gate(target, qubits):
    """Check if target is a standard gate."""
    # CNOT
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        
        # Check CNOT
        if _distance(cnot, target) < 1e-9:
            return "cnot"
        # Check SWAP
        if _distance(swap, target) < 1e-9:
            return "swap"
        # Check iSWAP
        if _distance(iswap, target) < 1e-9:
            return "iswap"
            
    if qubits == 3:
        toffoli = np.zeros((8,8), dtype=complex)
        for i in range(8):
            toffoli[i,i] = 1
        # Toffoli: flip bit 2 if bits 0 and 1 are 1
        # Basis: 000, 001, 010, 011, 100, 101, 110, 111
        # Indices: 0, 1, 2, 3, 4, 5, 6, 7
        # 110 (6) -> 111 (7)
        # 111 (7) -> 110 (6)
        toffoli[6,6] = 0
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        toffoli[7,7] = 0
        
        if _distance(toffoli, target) < 1e-9:
            return "toffoli"
            
    return None

def _get_fixed_gates(name, qubits):
    if name == "cnot":
        return [["cx", 0, 1]]
    if name == "swap":
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    if name == "iswap":
        # 3-CNOT decomposition
        return [
            ["u3", 0, math.pi/2, 0, math.pi],
            ["u3", 1, math.pi/2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, math.pi/2, -math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 1, 0],
            ["u3", 0, math.pi/2, 0, math.pi],
            ["u3", 1, math.pi/2, 0, math.pi]
        ]
    if name == "toffoli":
        # 6-CNOT decomposition
        return [
            ["u3", 2, math.pi/2, 0, math.pi],
            ["cx", 1, 2],
            ["u3", 2, math.pi/2, math.pi/2, -math.pi/2],
            ["cx", 0, 2],
            ["u3", 2, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 1, 2],
            ["u3", 2, math.pi/2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 1, math.pi/2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 1, math.pi/2, 0, math.pi]
        ]
    return None

def synthesize(problem):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for fixed gates
    gate_name = _is_fixed_gate(target, qubits)
    if gate_name:
        gates = _get_fixed_gates(gate_name, qubits)
        # Verify CNOT count
        cx_count = sum(1 for g in gates if g[0] == "cx")
        if cx_count <= budget:
            return gates
        # If budget is smaller (unlikely), fall through to optimization
    
    # Generic Synthesis
    # Determine CNOT ladder
    if qubits == 2:
        pairs = [(0, 1), (1, 0), (0, 1)]
        num_cx = 3
    else:
        pairs = [(0, 1), (1, 2), (0, 2)]
        # Use as many CNOTs as budget allows, but cap at a reasonable number for optimization speed
        # Standard 3-qubit decomp is ~12-15 CNOTs.
        num_cx = min(budget, 15)
        
    # Build ladder
    layout = []
    # Pre U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(num_cx):
        layout.append(["cx", *pairs[i % len(pairs)]])
        # Post U3s for this layer
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    # Final U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    # Count U3s
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(values):
        gates = circuit(values)
        U = _build(gates, qubits)
        return _distance(target, U)
        
    # Initial guess
    x0 = np.zeros(num_u3 * 3)
    
    # Optimization
    best_x = x0
    best_val = objective(x0)
    
    # Time check
    if time.time() - _START_TIME > _DEADLINE - 10:
        return circuit(best_x)
        
    try:
        # Use BFGS or L-BFGS-B
        # L-BFGS-B is good for large problems
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-12})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # If time remains, try a few random restarts or local search
    if time.time() - _START_TIME < _DEADLINE - 20:
        rng = random.Random(42)
        for _ in range(5):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            x_rand = rng.uniform(-math.pi, math.pi, size=num_u3 * 3)
            res = minimize(objective, x_rand, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-12})
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
                
    return circuit(best_x)

# EVOLVE-BLOCK-END
