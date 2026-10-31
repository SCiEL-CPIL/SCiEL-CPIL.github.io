    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # Matches C00
        [0, 1, 0, 1, 0, 0, 0],  # Wait, code has [0, 1, 0, 1, 0, 0, 0] for row 1?
        [0, 0, 1, 0, 1, 0, 0],  # Code has [0, 0, 1, 0, 1, 0, 0] for row 2?
        [1, -1, 1, 0, 0, 1, 0]  # Matches C11
    ], dtype=int)
