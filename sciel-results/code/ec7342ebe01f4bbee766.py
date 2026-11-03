    def _strassen_4x4x4():
        U = np.zeros((16, 49), dtype=np.float64)
        V = np.zeros((28, 49), dtype=np.float64)
        W = np.zeros((28, 49), dtype=np.float64)
        # ... rest of code same, but V and W are 28 rows.
    