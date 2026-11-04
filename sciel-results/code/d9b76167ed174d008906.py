def _strassen_4x4():
    U2, V2, W2 = _strassen_2x2()
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    return U4, V4, W4
