def _Q_eval(Q, y):
    """Evaluate Q(y) using Horner's method. Q is ascending coefficients."""
    result = Q[-1]
    for c in Q[-2::-1]:
        result = result * y + c
    return result
