<<<<<<< SEARCH
    # Generic synthesis
    return _generic_synthesize(target, qubits, budget, name)
=======
    # Generic synthesis
    if qubits == 3:
        result = _solve_3qubit_linear(target, budget)
        if result is not None:
            return result
    
    return _generic_synthesize_cd(target, qubits, budget, name)
>>>>>>> REPLACE
