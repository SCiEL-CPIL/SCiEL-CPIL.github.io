for j in range(k-1, -1, -1):
    coeffs[j] = (idx % 3) - 1
    idx //= 3
