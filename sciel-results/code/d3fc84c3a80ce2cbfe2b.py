    buffer = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    def is_valid(network):
        if not network:
            return False
        np.copyto(buffer, inputs)
        for i, j in network:
            min_vals = np.minimum(buffer[:, i], buffer[:, j])
            max_vals = np.maximum(buffer[:, i], buffer[:, j])
            buffer[:, i] = min_vals
            buffer[:, j] = max_vals
        return np.all(buffer[:, :-1] <= buffer[:, 1:])
