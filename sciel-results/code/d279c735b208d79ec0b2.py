"""Optimized uncertainty inequality via careful multi-start gradient optimization."""
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    We minimize the product of spatial and frequency variances (upper bound on C4).
    """
    # Constraint: g(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0
    # => a12 = -(a0 + 3*a4 + 105*a8) / 10395
    inv_10395 = 1.0 / 10395.0
    c0 = inv_10395
    c4 = 3.0 * inv_10395
    c8 = 105.0 * inv_10395

    # Norms: <He_n, He_n> = n!
    f = np.array([1.0, 24.0, 40320.0, 479001600.0])  # n=0,4,8,12

    # <x^2 He_n, He_n> = (2n+1) * n!
    x2 = np.array([1.0, 9*24.0, 17*40320.0, 25*479001600.0])

    # <He_n', He_n'> = n * n!
    kk = np.array([0.0, 4*24.0, 8*40320.0, 12*479001600.0])

    def compute_a12(a0, a4, a8):
        return -(c0 * a0 + c4 * a4 + c8 * a8)

    def objective_and_grad(v):
        a0, a4, a8 = v
        a12 = compute_a12(a0, a4, a8)
        a = np.array([a0, a4, a8, a12])

        N = a @ (f * a)
        if N < 1e-20:
            return 1e15, np.zeros(3)

        X = a @ (x2 * a)
        K = a @ (kk * a)

        val = (X * K) / (N * N)

        # da12/d(ai) for i in {0,4,8}
        da12 = np.array([-c0, -c4, -c8])

        # Derivatives with respect to a0, a4, a8 (accounting for a12 dependence)
        # dN/da_i = 2*(f[i]*a[i] + f[3]*a12*da12[i])
        # dX/da_i = 2*(x2[i]*a[i] + x2[3]*a12*da12[i])
        # dK/da_i = 2*(kk[i]*a[i] + kk[3]*a12*da12[i])

        dN = np.array([
            2*(f[0]*a0 + f[3]*a12*da12[0]),
            2*(f[1]*a4 + f[3]*a12*da12[1]),
            2*(f[2]*a8 + f[3]*a12*da12[2])
        ])
        dX = np.array([
            2*(x2[0]*a0 + x2[3]*a12*da12[0]),
            2*(x2[1]*a4 + x2[3]*a12*da12[1]),
            2*(x2[2]*a8 + x2[3]*a12*da12[2])
        ])
        dK = np.array([
            2*(kk[0]*a0 + kk[3]*a12*da12[0]),
            2*(kk[1]*a4 + kk[3]*a12*da12[1]),
            2*(kk[2]*a8 + kk[3]*a12*da12[2])
        ])

        # d(val)/da_i = (K*dX_i + X*dK_i - 2*val*dN_i) / N^2
        grad = (K * dX + X * dK - 2.0 * val * dN) / (N * N)

        return val, grad

    def obj_fn(v):
        return objective_and_grad(v)[0]

    def grad_fn(v):
        return objective_and_grad(v)[1]

    # Generate diverse starting points
    best_val = float('inf')
    best_x = None

    # Systematic grid + random starts
    starts = []
    # Grid in log-space for ratios
    for a0 in np.linspace(-2, 2, 9):
        for a4 in np.linspace(-2, 2, 9):
            for a8 in np.linspace(-2, 2, 9):
                starts.append(np.array([a0, a4, a8]))

    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(200):
        s = rng.standard_normal(3) * rng.uniform(0.1, 3.0)
        starts.append(s)

    # Add targeted starts based on structure
    for scale in [0.01, 0.1, 1.0, 10.0]:
        for sign in [-1, 1]:
            starts.append(sign * scale * np.array([1.0, 0.1, 0.01]))
            starts.append(sign * scale * np.array([1.0, -0.1, 0.01]))
            starts.append(sign * scale * np.array([0.1, 1.0, 0.01]))
            starts.append(sign * scale * np.array([0.01, 0.1, 1.0]))

    # Run optimization from each start
    for start in starts:
        try:
            res = minimize(obj_fn, start, jac=grad_fn, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Refine the best solution with tighter tolerances
    if best_x is not None:
        try:
            res = minimize(obj_fn, best_x, jac=grad_fn, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if best_x is None:
        best_x = np.array([0.0, 0.0, 0.0])

    return best_x

if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
