def _apply_comparator(state, i, j):
    a = state[i]
    b = state[j]
    state[i] = a & b
    state[j] = a | b
