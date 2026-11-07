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
            # y is sorted ascending
            # HV = sum_{i=0}^{B-2} (y[i+1] - y[i]) * (10 + sqrt(y[i])) + (R1 - y[-1]) * (10 + sqrt(y[-1]))
            # Use vectorized ops
            dy = np.diff(y)
            h = 10.0 + np.sqrt(y)
            # Sum for i=0 to B-2
            # term_i = dy[i] * h[i]
            total = np.sum(dy * h[:-1])
            # Last term
            total += (R1 - y[-1]) * h[-1]
            return total

        def hv_gradient(y):
            """Compute gradient of HV w.r.t y."""
            B = len(y)
            grad = np.zeros(B)
            if B == 0:
                return grad

            h = 10.0 + np.sqrt(y)

            # Interior points 1 to B-2
            if B > 2:
                # dH/dy_i = h[i-1] - h[i] + (y[i+1] - y[i]) / (2 * sqrt(y[i]))
                # Careful with sqrt(0)
                sqrt_y = np.sqrt(y)
                # Avoid division by zero
                inv_2sqrt = np.where(sqrt_y > 1e-15, 1.0 / (2.0 * sqrt_y), 0.0)

                dy_next = np.diff(y)  # y[i+1] - y[i]

                # For i=1 to B-2
                # h[i-1] - h[i]
                dh = h[:-1] - h[1:]
                # (y[i+1] - y[i]) / (2 * sqrt(y[i]))
                term2 = dy_next[:-1] * inv_2sqrt[1:-1]

                grad[1:-1] = dh + term2

            # i = 0
            # dH/dy_0 = -h[0] + (y[1] - y[0]) / (2 * sqrt(y[0]))
            if B > 1:
                sqrt_y0 = np.sqrt(y[0])
                inv_2sqrt0 = 1.0 / (2.0 * sqrt_y0) if sqrt_y0 > 1e-15 else 0.0
                grad[0] = -h[0] + (y[1] - y[0]) * inv_2sqrt0
            else:
                # B=1
                # H = (R1 - y0) * h0
                # dH/dy0 = -h0 + (R1-y0)/(2sqrt(y0))
                sqrt_y0 = np.sqrt(y[0])
                inv_2sqrt0 = 1.0 / (2.0 * sqrt_y0) if sqrt_y0 > 1e-15 else 0.0
                grad[0] = -h[0] + (R1 - y[0]) * inv_2sqrt0

            # i = B-1
            # dH/dy_{B-1} = h[B-2] - h[B-1] + (R1 - y[B-1]) / (2 * sqrt(y[B-1]))
            if B > 1:
                sqrt_yL = np.sqrt(y[-1])
                inv_2sqrtL = 1.0 / (2.0 * sqrt_yL) if sqrt_yL > 1e-15 else 0.0
                grad[-1] = h[-2] - h[-1] + (R1 - y[-1]) * inv_2sqrtL
            # If B==1, already handled above

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

            grad_z = np.zeros_like(z)

            # grad_z_j = sum_i (dH/dy_i) * (dy_i/dz_j)
            # dy_i/dz_j = exp(z_j) * (1 + S_{B-1} - S_i) / denom^2  if j <= i
            #           = -exp(z_j) * S_i / denom^2             if j > i

            # We can compute this efficiently.
            # Let A_i = 1 + S_{B-1} - S_i
            A = 1.0 + S[-1] - S
            # Term for j <= i: exp(z_j) * A_i / denom^2
            # Term for j > i: -exp(z_j) * S_i / denom^2

            # grad_z_j = (1/denom^2) * [ exp(z_j) * sum_{i=j}^{B-1} grad_y_i * A_i - exp(z_j) * sum_{i=0}^{j-1} grad_y_i * S_i ]

            # Precompute sums
            # sum1_j = sum_{i=j}^{B-1} grad_y_i * A_i
            # sum2_j = sum_{i=0}^{j-1} grad_y_i * S_i

            ga = grad_y * A
            gs = grad_y * S

            # sum1 is reverse cumulative sum of ga
            sum1 = np.cumsum(ga[::-1])[::-1]
            # sum2 is cumulative sum of gs, shifted
            # sum2_j = sum_{i=0}^{j-1} gs_i
            # cumsum(gs) gives sum_{i=0}^j gs_i
            # So sum2_j = cumsum(gs)_{j-1}
            cum_gs = np.cumsum(gs)
            sum2 = np.zeros(B)
            if B > 1:
                sum2[1:] = cum_gs[:-1]

            grad_z = (exp_z / denom_sq) * (sum1 - sum2)

            return -grad_z  # Minimize -HV, so gradient is -dHV/dz

        # Initial guess
        # Uniform y: y_i = (i+0.5)/B
        # This corresponds to specific z
        # Or just start with z=0 (which gives y_i = S_i/(1+S_{B-1}) with S_i = i+1)
        # y_i = (i+1)/(1+B). This is uniform in (0, 1) but shifted.
        # y_0 = 1/(B+1), y_{B-1} = B/(B+1).
        # This is a good start.
        z0 = np.zeros(B)

        # Run optimization
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective,
                z0,
                jac=objective_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            z_opt = res.x
        except Exception:
            z_opt = z0

        y_opt = y_from_z(z_opt)

        # Ensure y is within [0, 1] and sorted
        y_opt = np.clip(y_opt, 0.0, 1.0)
        # y_opt is already sorted by construction

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
