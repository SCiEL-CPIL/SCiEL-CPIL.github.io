"""Rewritten program for exact quantum circuit synthesis."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _build_fast(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Apply to qubit q
            # qubit 0 is MSB
            # We can build the operator more efficiently
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == q:
                    operator = np.kron(operator, single)
                else:
                    operator = np.kron(operator, eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT
            # Instead of full matrix mult, we can permute rows
            # But for small N, matrix mult is fine
            # Let's stick to the seed's method for CNOT as it's fast enough for N=4,8
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            # Swap rows where control is 1
            # The seed's method: matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
            # This effectively swaps the rows.
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _is_same_unitary(u1, u2, tol=1e-6):
    # Check if u1 and u2 are the same up to global phase
    # u1 = e^{i phi} u2
    # trace(u1^dagger u2) = e^{-i phi} trace(u2^dagger u2) = e^{-i phi} N
    # So |trace(u1^dagger u2)| should be N
    # And the phase should be consistent
    n = u1.shape[0]
    tr = np.trace(u1.conj().T @ u2)
    if abs(abs(tr) - n) > tol:
        return False
    # Check if u1 == e^{i phi} u2
    if abs(tr) < 1e-9:
        return False
    phi = np.angle(tr)
    # Check a few elements
    for i in range(min(5, n)):
        for j in range(min(5, n)):
            if abs(u1[i, j] - np.exp(1j * phi) * u2[i, j]) > tol:
                return False
    return True


def _get_standard_gates(qubits):
    """Returns a list of (name, target_matrix, gates) for standard gates."""
    gates_list = []
    if qubits == 2:
        # CNOT
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        gates_list.append(("CNOT", cnot, [["cx", 0, 1]]))
        
        # SWAP
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        gates_list.append(("SWAP", swap, [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]))
        
        # iSWAP
        # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
        iswap = np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
        # Decomposition for iSWAP with 3 CNOTs
        # u3(pi/2, 0, pi/2) on q0
        # u3(pi/2, pi/2, -pi/2) on q1
        # cx(0,1)
        # u3(pi/2, 0, pi/2) on q0? 
        # Let's use a known decomposition:
        # q0: u3(pi/2, 0, pi/2)
        # q1: u3(pi/2, pi/2, -pi/2)
        # cx(0,1)
        # q0: u3(pi/2, 0, pi/2)
        # q1: u3(pi/2, pi/2, -pi/2)
        # cx(1,0)
        # This is 2 CNOTs? No, iSWAP needs 3 CNOTs.
        # Let's just use the optimizer for iSWAP if it's not exact, but it is a standard gate.
        # Standard iSWAP decomposition:
        # u3(pi/2, 0, pi/2) on q0
        # u3(pi/2, pi/2, -pi/2) on q1
        # cx(0,1)
        # u3(pi/2, 0, pi/2) on q0
        # u3(pi/2, pi/2, -pi/2) on q1
        # cx(1,0)
        # This is 2 CNOTs. Is iSWAP 2 CNOTs?
        # Yes, iSWAP can be done with 2 CNOTs if we allow arbitrary single-qubit gates.
        # Let's verify the matrix.
        # Actually, let's just add it to the list and let the optimizer handle it if the hardcoded one is wrong.
        # But we want exact.
        # Let's use the 3-CNOT version to be safe? No, 2 is better if correct.
        # Let's assume the standard 3-CNOT ladder is used for generic 2-qubit.
        
    elif qubits == 3:
        # Toffoli
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        # Standard 7-CNOT Toffoli
        # q0: u3(pi/2, 0, 0)
        # cx(0,1)
        # q1: u3(pi/4, 0, 0)
        # cx(0,2)
        # q2: u3(-pi/4, 0, 0)
        # cx(0,2)
        # q1: u3(pi/4, 0, 0)
        # cx(0,1)
        # q0: u3(-pi/2, 0, 0)
        # q1: u3(-pi/2, 0, 0)
        # cx(1,2)
        # q1: u3(pi/2, 0, 0)
        # This is 7 CNOTs.
        
        # Fredkin (Controlled-SWAP)
        fredkin = np.eye(8, dtype=complex)
        # Swap q1 and q2 if q0 is 1
        # Indices: 000, 001, 010, 011, 100, 101, 110, 111
        # 010 (2) <-> 001 (1)? No.
        # If q0=1 (indices 4-7), swap q1 and q2.
        # 100 (4) <-> 100 (4) (q1=0, q2=0)
        # 101 (5) <-> 110 (6) (q1=0, q2=1 <-> q1=1, q2=0)
        # 110 (6) <-> 101 (5)
        # 111 (7) <-> 111 (7)
        fredkin[5, 5] = 0
        fredkin[5, 6] = 1
        fredkin[6, 5] = 1
        fredkin[6, 6] = 0
        # Fredkin decomposition with 3 CNOTs
        # q0: u3(pi/2, 0, 0)
        # cx(0,1)
        # q1: u3(-pi/4, 0, 0)
        # cx(1,2)
        # q2: u3(pi/4, 0, 0)
        # cx(0,2)
        # q2: u3(-pi/4, 0, 0)
        # cx(0,1)
        # q0: u3(-pi/2, 0, 0)
        # q1: u3(pi/2, 0, 0)
        # This is 4 CNOTs?
        # Let's just use the optimizer for Fredkin if not exact.

    return gates_list


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Check for standard gates
    std_gates = _get_standard_gates(qubits)
    for name, mat, gates in std_gates:
        if _is_same_unitary(target, mat):
            return gates

    # Fallback: Optimization
    # Define a standard topology
    # For 2 qubits: 3 CNOTs
    # For 3 qubits: 7 CNOTs
    if qubits == 2:
        layout = [
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]
        ]
    else:
        # 7 CNOT Toffoli-like topology
        layout = [
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0],
            ["cx", 0, 2],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0],
            ["cx", 0, 2],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0],
            ["cx", 1, 2],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["u3", 2, 0, 0, 0]
        ]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    x0 = np.zeros(n_params)
    
    def objective(x):
        if _check_deadline():
            return 1.0
        gates = circuit(x)
        u = _build_fast(gates, qubits)
        return _distance(target, u)

    # Use Nelder-Mead for robustness
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        best_x = res.x
    except Exception:
        best_x = x0

    return circuit(best_x)
# EVOLVE-BLOCK-END
