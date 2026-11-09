_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = np.zeros((1 << N, N), dtype=np.int8)
        for i in range(N):
            _INPUTS[:, i] = (np.arange(1 << N) >> i) & 1
    return _INPUTS

def _verify(network):
    inputs = _get_inputs().copy() # Must copy because we modify it
    for i, j in network:
        a = inputs[:, i]
        b = inputs[:, j]
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    return np.all(inputs[:, :-1] <= inputs[:, 1:])
