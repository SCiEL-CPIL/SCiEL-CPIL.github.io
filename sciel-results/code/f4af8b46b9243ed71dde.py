W[4*k, base + r] = W_local[0][r]      # C00 ✓
W[4*k+2, base + r] = W_local[1][r]    # Should be C10, but assigns C01 ✗
W[4*k+1, base + r] = W_local[2][r]    # Should be C01, but assigns C10 ✗
W[4*k+3, base + r] = W_local[3][r]    # C11 ✓
