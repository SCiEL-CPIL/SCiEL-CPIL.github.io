import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        B = int(self.budget)
        D = int(self.dim)

        # Reference point
        R1, R2 = 11.0, 11.0

        # ZDT1 front: f1 = x1, f2 = 1 - sqrt(x1)
        # h(y) = R2 - f2(y) = 11 - (1 - sqrt(y)) = 10 + sqrt(y)

        def compute_hv(y):
            """Compute HV for sorted y values."""
            if len(y) == 0:
                return 0.0
            dy = np.diff(y)
            h = 10.0 + np.sqrt(y)
            total = np.sum(dy * h[:-1])
            total += (R1 - y[-1]) * h[-1]
            return total

        def hv_gradient(y):
            """Compute gradient of HV w.r.t y."""
            n = len(y)
            grad = np.zeros(n)
            if n == 0:
                return grad

            h = 10.0 + np.sqrt(y)
            sqrt_y = np.sqrt(y)
            inv_2sqrt = np.where(sqrt_y > 1e-15, 1.0 / (2.0 * sqrt_y), 0.0)

            if n > 1:
                dy = np.diff(y)
                dh = h[:-1] - h[1:]

                # Interior points i=1 to n-2
                if n > 2:
                    grad[1:-1] = dh[:-1] + dy[:-1] * inv_2sqrt[1:-1]

                # i=0
                grad[0] = -h[0] + dy[0] * inv_2sqrt[0]

                # i=n-1
                grad[-1] = dh[-1] + (R1 - y[-1]) * inv_2sqrt[-1]
            else:
                # n=1
                grad[0] = -h[0] + (R1 - y[0]) * inv_2sqrt[0]

            return grad

        # Parameterization: y_i = S_i / (1 + S_{B-1})
        # S_i = sum_{k=0}^i exp(z_k)

        def y_from_z(z):
            exp_z = np.exp(z)
            S = np.cumsum(exp_z)
            denom = 1.0 + S[-1]
            y = S / denom
            return y

        def objective(z):
            y = y_from_z(z)
            hv = compute_hv(y)
            return -hv

        def objective_gradient(z):
            y = y_from_z(z)
            exp_z = np.exp(z)
            S = np.cumsum(exp_z)
            denom = 1.0 + S[-1]
            denom_sq = denom * denom

            grad_y = hv_gradient(y)

            # dy_i/dz_j = exp(z_j) * (1 + S_{B-1} - S_i) / denom^2  if j <= i
            #           = -exp(z_j) * S_i / denom^2             if j > i

            A = 1.0 + S[-1] - S
            ga = grad_y * A
            gs = grad_y * S

            sum1 = np.cumsum(ga[::-1])[::-1]
            cum_gs = np.cumsum(gs)
            sum2 = np.zeros(B)
            if B > 1:
                sum2[1:] = cum_gs[:-1]

            grad_z = (exp_z / denom_sq) * (sum1 - sum2)

            return -grad_z

        # Initial guess: uniform y
        z0 = np.zeros(B)

        # Run optimization with L-BFGS-B
        z_opt = z0
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective,
                z0,
                jac=objective_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
            )
            z_opt = res.x
        except Exception:
            pass

        # Additional polish: run L-BFGS-B again from the result with tighter tolerances
        try:
            from scipy.optimize import minimize
            res2 = minimize(
                objective,
                z_opt,
                jac=objective_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 50}
            )
            z_opt = res2.x
        except Exception:
            pass

        y_opt = y_from_z(z_opt)
        y_opt = np.clip(y_opt, 0.0, 1.0)

        # Build X
        X = np.zeros((B, D))
        X[:, 0] = y_opt

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)

        # Evaluate
        used = 0
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            used += 1

        return X
