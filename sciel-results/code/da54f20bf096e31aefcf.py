<<<<<<< SEARCH
    if qubits == 2:
        return _decompose_2qubit_kak(target)
    elif qubits == 3:
        return _decompose_3qubit_ladder(target, budget)
=======
    if qubits == 2:
        return _decompose_2qubit_kak(target, budget)
    elif qubits == 3:
        return _decompose_3qubit_ladder(target, budget)
>>>>>>> REPLACE
