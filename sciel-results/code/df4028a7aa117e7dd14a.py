def _is_close_to(target, gates, qubits):
    matrix = _build_fast(gates, qubits)
    return _distance(target, matrix) < 1e-6

def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check known gates
    if qubits == 2:
        if _is_close_to(target, [["cx", 0, 1]], 2):
            return [["cx", 0, 1]]
        if _is_close_to(target, [["cx", 1, 0]], 2):
            return [["cx", 1, 0]]
        if _is_close_to(target, _synthesize_swap(problem), 2):
            return _synthesize_swap(problem)
        # iSWAP check?
        # iSWAP is not in the "known" list in the prompt's explicit list?
        # "CNOT, iSWAP, SWAP, Toffoli"
        # I will add an iSWAP check.
        iswap_gates = [
            ["u3", 0, math.pi/2, -math.pi/2, math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, math.pi/2, 0],
            ["u3", 1, math.pi/2, 0, math.pi/2],
            ["cx", 1, 0],
            ["u3", 0, math.pi/2, 0, math.pi/2],
            ["u3", 1, math.pi/2, math.pi/2, 0],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, math.pi/2, 0],
            ["u3", 1, math.pi/2, 0, math.pi/2]
        ]
        if _is_close_to(target, iswap_gates, 2):
            return iswap_gates
            
    if qubits == 3 and name == "Toffoli":
        if _is_close_to(target, _synthesize_toffoli(problem), 3):
            return _synthesize_toffoli(problem)
            
    # Generic optimization
    return _optimize_generic(problem, target, budget)[0]
