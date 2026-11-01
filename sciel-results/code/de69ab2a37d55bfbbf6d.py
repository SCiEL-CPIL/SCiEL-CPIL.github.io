        def _penalty_energy_and_grad(X, lam):
            # X is (N, 3)
            # Energy
            energy, grad = _log_energy_and_grad(X)
            # Penalty
            norms_sq = np.sum(X*X, axis=1)
            pen = lam * np.sum((norms_sq - 1.0)**2)
            # Grad of penalty: d/dx_i lam * (|x_i|^2 - 1)^2 = lam * 2*(|x_i|^2 - 1) * 2*x_i = 4*lam*(|x_i|^2 - 1)*x_i
            pen_grad = 4.0 * lam * (norms_sq - 1.0)[:, np.newaxis] * X
            return energy + pen, grad + pen_grad
        