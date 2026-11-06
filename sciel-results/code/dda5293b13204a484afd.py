"""Direct synthesis: exact decompositions for standard gates + fast Nelder-Mead for generic unitaries."""
import time
import math
import random

import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    matrix = np.eye(dim, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _iswap_gates_exact():
    """iSWAP on qubits (0,1): |01> -> i|10>, |10> -> i|01>. 1-CNOT decomposition."""
    # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Decomposition: (Ry(π/4)⊗Ry(-π/4)) · CNOT(0,1) · (Ry(-π/4)⊗Ry(π/4)) · CNOT(0,1)
    # Wait, that's 2 CNOTs. For 1 CNOT:
    # iSWAP = U1 · CNOT(0,1) · U0
    # Let me derive: After CNOT(0,1): |01>->|11>, |10>->|10>
    # We want: |01>->i|10>, |10>->i|01>
    # So before CNOT: |01> should become something that CNOT maps to |10> with phase i
    # CNOT(0,1): |ab> -> |a, a⊕b>
    # |01> -> |0, 0⊕1> = |01> (unchanged) -- wait that's wrong
    # CNOT(0,1) with q0 as control: |q0 q1> -> |q0, q0⊕q1>
    # |01> -> |0, 0⊕1> = |01>
    # |10> -> |1, 1⊕0> = |11>
    # Hmm, so CNOT(0,1) doesn't swap. Let me reconsider.
    # Actually for iSWAP we need to map |01>->i|10> and |10>->i|01>
    # Using 1 CNOT: iSWAP = (Rz(π/4)⊗Rz(-π/4)) · (Ry(π/2)⊗Ry(π/2)) · CNOT(0,1) · (Ry(π/2)⊗Ry(π/2)) · (Rz(-π/4)⊗Rz(π/4))
    # Let me just use a verified 1-CNOT decomposition from literature.
    # iSWAP = H⊗H · CNOT(0,1) · (Ry(π/4)⊗Ry(-π/4)) · CNOT(0,1) · H⊗H  -- still 2 CNOTs
    # For 1 CNOT, we can use:
    # iSWAP = (Ry(π/4)⊗Ry(-π/4)) · CNOT(0,1) · (Ry(-π/4)⊗Ry(π/4)) · CNOT(0,1)  -- 2 CNOTs
    # The minimum CNOT count for iSWAP is 1. Let me look this up properly.
    # Actually, iSWAP can be decomposed with 1 CNOT:
    # iSWAP = U1 · CNOT · U0
    # where U0 and U1 are single-qubit gates.
    # From Nielsen & Chuang or similar:
    # iSWAP = (Ry(π/4)⊗Ry(-π/4)) · CNOT(0,1) · (Ry(-π/4)⊗Ry(π/4)) · CNOT(0,1)
    # This uses 2 CNOTs. For 1 CNOT, we need a different approach.
    # Let me just verify: the minimum CNOT count for iSWAP is indeed 1.
    # iSWAP = diag(1, i, -i, 1) · SWAP
    # SWAP needs 3 CNOTs, so that's not helpful.
    # Actually, I recall that iSWAP requires 1 CNOT. Let me construct it:
    # We want: |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
    # Using CNOT(0,1): |00>->|00>, |01>->|01>, |10>->|11>, |11>->|11>
    # So after CNOT, we have: |00>, |01>, |11>, |11> -- not quite right
    # Let me just use a known working decomposition with 1 CNOT.
    # From various sources, iSWAP with 1 CNOT:
    # Apply: Ry(π/4) on q0, Ry(-π/4) on q1
    # CNOT(0,1)
    # Apply: Ry(-π/4) on q0, Ry(π/4) on q1
    # Then add phase corrections.
    # Let me just verify numerically in the code.
    # For now, return a placeholder that will be verified.
    s = 1.0 / math.sqrt(2)
    gates = [
        ["u3", 0, math.pi / 4, 0, 0],   # Ry(π/4) on q0
        ["u3", 1, -math.pi / 4, 0, 0],  # Ry(-π/4) on q1
        ["cx", 0, 1],
        ["u3", 0, -math.pi / 4, 0, 0],  # Ry(-π/4) on q0
        ["u3", 1, math.pi / 4, 0, 0],   # Ry(π/4) on q1
        ["u3", 0, 0, 0, math.pi / 2],   # phase correction on q0
        ["u3", 1, 0, 0, -math.pi / 2],  # phase correction on q1
    ]
    return gates


def _toffoli_gates_exact():
    """Toffoli gate (CCNOT) with controls q0,q1 and target q2. 6 CNOTs."""
    # Standard 6-CNOT Toffoli decomposition
    # From Nielsen & Chuang or similar
    t = math.pi / 4
    gates = [
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["u3", 0, math.pi / 2, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
        ["cx", 0, 1],
        ["u3", 1, -t, 0, 0],
        ["cx", 2, 1],
        ["u3", 1, t, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, -math.pi / 2, 0, 0],
        ["u3", 1, -math.pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
    ]
    # Count CNOTs: cx at indices 1,3,5,7,11,13,15,19,21,23,25 = 11 CNOTs. Still too many!
    # The standard Toffoli with 6 CNOTs is different. Let me use the correct one.
    # From Barenco et al. or standard references:
    # Toffoli = (I⊗H⊗I) · CNOT(1,2) · (I⊗Ry(π/4)⊗I) · CNOT(0,2) · (I⊗Ry(-π/4)⊗I) · CNOT(1,2) · (I⊗Ry(π/4)⊗I) · CNOT(0,2) · (I⊗Ry(-π/4)⊗I) · (I⊗H⊗I)
    # Wait, that's still 4 CNOTs in that part plus 2 more = 6? Let me recount.
    # Actually, the standard 6-CNOT Toffoli is:
    gates = [
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["u3", 0, math.pi / 2, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
        ["cx", 0, 1],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["cx", 2, 1],
        ["u3", 1, math.pi / 4, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, -math.pi / 2, 0, 0],
        ["u3", 1, -math.pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],
    ]
    # I keep getting 11 CNOTs. The issue is that the standard Toffoli decomposition I'm using is not the 6-CNOT one.
    # Let me look up the actual 6-CNOT Toffoli.
    # From https://en.wikipedia.org/wiki/Toffoli_gate#Quantum_circuit
    # The 6-CNOT Toffoli uses:
    # 1. H on target
    # 2. CNOT(control2, target)
    # 3. T† on target
    # 4. CNOT(control1, target)
    # 5. T on target
    # 6. CNOT(control2, target)
    # 7. T† on target
    # 8. CNOT(control1, target)
    # 9. T on target
    # 10. H on target
    # 11. CNOT(control1, control2)
    # 12. T on control2
    # 13. T† on control1
    # 14. CNOT(control1, control2)
    # 15. T on control1
    # 16. T on control2
    # That's 6 CNOTs! Let me implement this.
    # T gate = R_z(π/4) = u3(0, 0, π/4) ... wait, T = diag(1, e^{iπ/4})
    # In u3 convention: u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
    # T = [[1, 0], [0, e^{iπ/4}]] = u3(0, 0, π/4)? Let's check:
    # u3(0, 0, π/4) = [[1, 0], [0, e^{iπ/4}]] -- yes!
    # T† = [[1, 0], [0, e^{-iπ/4}]] = u3(0, 0, -π/4)
    # H = u3(π/2, 0, π)
    gates = [
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
        ["cx", 1, 2],                         # CNOT(1,2)
        ["u3", 2, 0, 0, -math.pi / 4],       # T† on q2
        ["cx", 0, 2],                         # CNOT(0,2)
        ["u3", 2, 0, 0, math.pi / 4],        # T on q2
        ["cx", 1, 2],                         # CNOT(1,2)
        ["u3", 2, 0, 0, -math.pi / 4],       # T† on q2
        ["cx", 0, 2],                         # CNOT(0,2)
        ["u3", 2, 0, 0, math.pi / 4],        # T on q2
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
        ["cx", 0, 1],                         # CNOT(0,1)
        ["u3", 1, 0, 0, math.pi / 4],        # T on q1
        ["u3", 0, 0, 0, -math.pi / 4],       # T† on q0
        ["cx", 0, 1],                         # CNOT(0,1)
        ["u3", 0, 0, 0, math.pi / 4],        # T on q0
        ["u3", 1, 0, 0, math.pi / 4],        # T on q1
    ]
    # Count CNOTs: 6 CNOTs. 
    return gates


def _known_decompositions(problem):
    """Try to return known exact decompositions for standard gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # CNOT
    if name == "CNOT" and qubits == 2 and budget >= 1:
        return [["cx", 0, 1]]
    
    # SWAP: 3 CNOTs
    if name == "SWAP" and qubits == 2 and budget >= 3:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    
    # iSWAP: 1 CNOT
    if name == "iSWAP" and qubits == 2 and budget >= 1:
        gates = _iswap_gates_exact()
        # Verify it's correct
        dist = _distance(_target(problem), _build_fast(gates, qubits))
        if dist < 1e-8:
            return gates
        return None
    
    # Toffoli: 6 CNOTs
    if name == "Toffoli" and qubits == 3 and budget >= 6:
        gates = _toffoli_gates_exact()
        # Verify it's correct
        dist = _distance(_target(problem), _build_fast(gates, qubits))
        if dist < 1e-8:
            return gates
        return None
    
    return None


def _build_gates(cnots_layout, values):
    """Build gate list from CNOT layout and parameter values."""
    gates = []
    k = 0
    for gate in cnots_layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _nelder_mead(target, qubits, cnots_layout, initial_values, time_limit, max_iter=200):
    """Nelder-Mead optimization on single-qubit parameters."""
    n = len(initial_values)
    if n == 0:
        return list(initial_values), _distance(target, _build_fast(_build_gates(cnots_layout, initial_values), qubits))
    
    # Initialize simplex
    simplex = [list(initial_values)]
    for i in range(n):
        perturbed = list(initial_values)
        step = 0.1 if abs(perturbed[i]) < 1e-6 else 0.1 * abs(perturbed[i])
        perturbed[i] += step
        simplex.append(perturbed)
    
    def objective(vals):
        gates = _build_gates(cnots_layout, vals)
        return _distance(target, _build_fast(gates, qubits))
    
    # Evaluate initial simplex
    f_values = [objective(s) for s in simplex]
    
    for iteration in range(max_iter):
        if time.time() - _start_time > time_limit:
            break
        
        # Sort simplex by objective value
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if f_values[0] < 1e-12:
            break
        if abs(f_values[0] - f_values[-1]) < 1e-12:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        alpha = 1.0
        reflected = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        f_reflected = objective(reflected)
        
        if f_values[0] <= f_reflected < f_values[n - 1]:
            simplex[n] = reflected
            f_values[n] = f_reflected
        elif f_reflected < f_values[0]:
            # Expansion
            gamma = 2.0
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[n] = expanded
                f_values[n] = f_expanded
            else:
                simplex[n] = reflected
                f_values[n] = f_reflected
        else:
            # Contraction
            rho = 0.5
            contracted = [centroid[j] + rho * (simplex[n][j] - centroid[j]) for j in range(n)]
            f_contracted = objective(contracted)
            if f_contracted < f_values[n]:
                simplex[n] = contracted
                f_values[n] = f_contracted
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    f_values[i] = objective(simplex[i])
    
    # Find best
    best_idx = min(range(n + 1), key=lambda i: f_values[i])
    return simplex[best_idx], f_values[best_idx]


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    dim = 2 ** qubits
    time_limit = _DEADLINE - 5.0  # 5s safety margin
    
    # First try known decompositions
    known = _known_decompositions(problem)
    if known is not None:
        dist = _distance(target, _build_fast(known, qubits))
        if dist < 1e-8:
            return known
    
    # Search approach: generate varied candidate circuits with Nelder-Mead
    best_gates = None
    best_dist = 1.0  # worst case
    
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Try multiple layouts with Nelder-Mead
    num_layouts = 6 if qubits <= 2 else 3
    rng = random.Random(12345 + qubits)
    
    for layout_idx in range(num_layouts):
        if time.time() - _start_time > time_limit:
            break
        
        # Generate a CNOT layout
        layout = []
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        for i in range(budget):
            if i < len(pairs):
                c, t = pairs[i]
            else:
                c, t = rng.choice(pairs)
            layout.append(["cx", c, t])
            if i < budget - 1:
                layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3
        
        # Try multiple initializations
        for init_idx in range(3):
            if time.time() - _start_time > time_limit:
                break
            
            if init_idx == 0:
                initial = [0.0] * n_params
            elif init_idx == 1:
                initial = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            else:
                initial = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
            
            values, dist = _nelder_mead(target, qubits, layout, initial, time_limit, max_iter=150)
            
            if dist < best_dist:
                best_dist = dist
                best_gates = _build_gates(layout, values)
    
    if best_gates is None:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
