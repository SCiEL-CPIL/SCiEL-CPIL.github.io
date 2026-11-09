    def evaluate(p):
        U_built = np.eye(4, dtype=complex)
        idx = 0
        for i in range(6):
            for q in range(2):
                theta = p[idx]; phi = p[idx+1]; lam = p[idx+2]
                U_built = np.kron(U_built if q == 0 else np.eye(2, dtype=complex),
                                  _u3_matrix(theta, phi, lam)) if q == 0 else \
                          np.kron(np.eye(2, dtype=complex), _u3_matrix(theta, phi, lam)) @ U_built
                idx += 3
            if i < 5:
                U_built = CX @ U_built
        return _distance(V, U_built)
