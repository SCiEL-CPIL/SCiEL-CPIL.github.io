U = np.zeros((16, rank), dtype=np.float64)
V = np.zeros((28, rank), dtype=np.float64)
W = np.zeros((28, rank), dtype=np.float64)

U[:, :49] = U1
V[:16, :49] = V1
W[:16, :49] = W1

U[:, 49:] = U2
V[16:, 49:] = V2
W[16:, 49:] = W2
