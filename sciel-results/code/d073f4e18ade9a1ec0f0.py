<<<<<<< SEARCH
def _optimize_generic(problem, max_time=20.0):
    """Optimize a generic unitary using coordinate descent."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build a fixed CNOT layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += ["u3"] * qubits
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += ["u3"] * qubits
    
    # Count u3 slots
    n_u3 = sum(1 for g in layout if g == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate == "u3":
                gates.append(["u3", layout.index(gate) % qubits, values[k], values[k+1], values[k+2]])
                # This is wrong - need to track which qubit
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Better: build layout with qubit info
    layout2 = []
    for layer in range(budget):
        for q in range(qubits):
            layout2.append(["u3", q, 0.0, 0.0, 0.0])
        layout2.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout2.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_u3 = sum(1 for g in layout2 if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout2:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with zeros
    best = [0.0] * n_params
    best_dist = _distance(target, _build(circuit(best), qubits))
    
    start = time.time()
    sweep = 0
    while time.time() - start < max_time and _time_remaining() > 5.0:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if time.time() - start > max_time or _time_remaining() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                val = _distance(target, _build(circuit(values), qubits))
                if val < best_dist:
                    best, best_dist = values, val
                    improved = True
        sweep += 1
        if not improved and sweep > 5:
            break
    
    return circuit(best)
=======
def _build_fast(params, qubits, cnot_ctrls, cnot_tgts, u3_qubits):
    """Fast circuit builder for optimization. params is flat [theta,phi,lam] per u3 slot."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    
    # Pre-compute CNOT permutation indices
    cnot_perms = []
    for ctrl, tgt in zip(cnot_ctrls, cnot_tgts):
        index = np.arange(n)
        bit = (index >> (qubits - 1 - ctrl)) & 1
        perm = index ^ (bit << (qubits - 1 - tgt))
        cnot_perms.append(perm)
    
    # Pre-compute single-qubit operator templates
    # For each qubit position, precompute the kron structure
    # We'll build incrementally
    
    n_u3 = len(u3_qubits)
    n_cnot = len(cnot_ctrls)
    
    # Track position in params
    param_idx = 0
    cnot_idx = 0
    
    # Build the circuit by alternating u3 and cx
    # We need to know the order: u3 slots and cnot slots interleaved
    # Actually, we need the full layout. Let's pass it differently.
    
    # Rebuild: we need the sequence of operations
    # Let me restructure: pass a list of ops where each op is either
    # ('u3', qubit, param_offset) or ('cx', ctrl, tgt)
    pass  # This function will be restructured below
    
    return mat


def _optimize_generic(problem, max_time=20.0):
    """Optimize a generic unitary using scipy L-BFGS-B."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build a fixed CNOT layout: alternate CNOT pairs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout: u3 on all qubits, then CNOT, repeat budget times, then final u3
    # Store as list of operations
    ops = []  # each element: ('u3', qubit) or ('cx', ctrl, tgt)
    for layer in range(budget):
        for q in range(qubits):
            ops.append(('u3', q))
        ops.append(('cx', pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
    for q in range(qubits):
        ops.append(('u3', q))
    
    # Count u3 slots
    u3_indices = [i for i, op in enumerate(ops) if op[0] == 'u3']
    u3_qubits = [ops[i][1] for i in u3_indices]
    n_u3 = len(u3_indices)
    n_params = 3 * n_u3
    
    # Pre-compute CNOT permutations
    cnot_ops = [(op[1], op[2]) for op in ops if op[0] == 'cx']
    cnot_perms = []
    for ctrl, tgt in cnot_ops:
        index = np.arange(n)
        bit = (index >> (qubits - 1 - ctrl)) & 1
        perm = index ^ (bit << (qubits - 1 - tgt))
        cnot_perms.append(perm)
    
    def _build_from_params(params):
        """Build unitary from flat parameter array."""
        mat = np.eye(n, dtype=complex)
        param_idx = 0
        cnot_idx = 0
        
        for op in ops:
            if op[0] == 'u3':
                q = op[1]
                theta = params[param_idx]
                phi = params[param_idx + 1]
                lam = params[param_idx + 2]
                param_idx += 3
                
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                single = np.array([[c, -np.exp(1j * lam) * s],
                                   [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                
                # Apply single-qubit gate on qubit q
                # Build the full operator via kron
                operator = np.array([[1]], dtype=complex)
                eye = np.eye(2, dtype=complex)
                for k in range(qubits):
                    operator = np.kron(operator, single if k == q else eye)
                mat = operator @ mat
            else:
                perm = cnot_perms[cnot_idx]
                cnot_idx += 1
                mat = mat[perm, :]
        
        return mat
    
    def _objective(params):
        if _time_remaining() < 2.0:
            return 1.0  # Penalize to stop
        U = _build_from_params(params)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / n)
    
    # Initialize with zeros
    x0 = np.zeros(n_params)
    best_dist = _objective(x0)
    
    start = time.time()
    
    # Use scipy L-BFGS-B
    try:
        result = _scipy_minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                     'disp': False}
        )
        if result.fun < best_dist:
            best_dist = result.fun
            x0 = result.x
    except Exception:
        pass
    
    # Check time and maybe do a second round if time allows
    if _time_remaining() > max_time * 0.5:
        try:
            result2 = _scipy_minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                         'disp': False}
            )
            if result2.fun < best_dist:
                best_dist = result2.fun
                x0 = result2.x
        except Exception:
            pass
    
    # Convert back to gate list
    gates = []
    param_idx = 0
    for op in ops:
        if op[0] == 'u3':
            q = op[1]
            gates.append(['u3', q, float(x0[param_idx]), float(x0[param_idx + 1]), float(x0[param_idx + 2])])
            param_idx += 3
        else:
            gates.append(['cx', op[1], op[2]])
    
    return gates
>>>>>>> REPLACE

