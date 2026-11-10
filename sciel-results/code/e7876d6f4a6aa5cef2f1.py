def _objective_scalar(coeffs):
    val, _ = _objective_and_grad(coeffs)
    return val
