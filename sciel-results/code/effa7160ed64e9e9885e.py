def _build_laderman_4x4():
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    # Helper
    def set_block(mat, bi, bj, coeffs, term):
        for ii in range(2):
            for jj in range(2):
                row = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                mat[row, term] = coeffs[ii, jj]

    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))
    # ... (I will fill in the correct 20 terms)
