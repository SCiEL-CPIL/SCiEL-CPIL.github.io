val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
err += abs(val - 1.0)
