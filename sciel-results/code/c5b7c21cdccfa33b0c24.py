"""Circle packing in perimeter-4 rectangle: thin-rectangle single-row design with smooth optimization."""
import numpy as np
from scipy.optimize import minimize

N = 21
_EPS = 1e-11
_DEADLINE = 150.0
_MARGIN = 12.0


def _row_value(xs, r):
    """Value = sum of radii for a 1-row packing with centers xs (sorted) and radius r."""
    if xs is None or len(xs) < N:
        return 0.0
    xs = np.asarray(xs, dtype=float)
    if np.any(np.isnan(xs)):
        return 0.0
    xs = np.sort(xs)
    # wall clearance at both ends
    left = xs[0] - r
    right = (2.0 - xs[-1]) - r
    inner = np.diff(xs) - 2.0 * r
    if len(inner):
        minsep = float(min(left, right, inner.min()))
    else:
        minsep = float(min(left, right))
    if minsep < -1e-6:
        return 0.0
    return N * r


def construct():
    import time
    t0 = time.time()
    remaining = _DEADLINE - _MARGIN

    # Base analytic solution: single row of 21 equal circles, thin rectangle.
    # H = 2r, W = 2 - H, circles centered at x = r + i*2r.
    r0 = 1.0 / 22.0
    xs0 = r0 + np.arange(N) * 2.0 * r0
    base_val = _row_value(xs0, r0)

    best_val = base_val
    best_xs = xs0.copy()

    # Smooth surrogate: maximize the soft-min of the (N+1) separation quantities
    # (two walls + N-1 inner gaps).  f(x) = -softmin(sep)  (minimize f).
    # sep_k are affine in x (hence f is convex), so L-BFGS-B is robust.
    def sep_from_xs(x):
        x = np.sort(x)
        s = np.empty(N + 1, dtype=float)
        s[0] = x[0] - r0
        s[1:-1] = np.diff(x) - 2.0 * r0
        s[-1] = (2.0 - x[-1]) - r0
        return s

    def f(x):
        s = sep_from_xs(x)
        m = s.min()
        # soft-min (log-sum-exp of negatives), beta large -> sharp
        z = -beta * s
        z = z - z.max()
        return -m - (1.0 / beta) * np.log(np.sum(np.exp(z)))

    def grad_f(x):
        s = sep_from_xs(x)
        m = s.min()
        z = -beta * s
        z = z - z.max()
        e = np.exp(z)
        w = e / e.sum()
        # d softmin/d s_i = -w_i ; f = -m - softmin; df/d s_i = w_i
        # s_0 = x_0 - r        -> ds_0/dx_0 = 1
        # s_k (1..N-1) = x_k - x_{k-1} - 2r -> ds/dx_{k-1}=-1, ds/dx_k=+1
        # s_{N-1} = 2 - x_{N-1} - r -> ds/dx_{N-1} = -1
        g = np.zeros(N, dtype=float)
        g[0] += w[0]
        for k in range(1, N - 1):
            g[k - 1] -= w[k]
            g[k] += w[k]
        g[N - 1] += w[N - 1]
        # note: f = -m - softmin, but the -m term is constant w.r.t x (min index fixed
        # locally); we ignore it (it is 0 a.e. and its subgradient is absorbed).
        return g

    # Use a few annealed beta values for a smooth->sharp polish.
    betas = [8.0, 20.0, 60.0, 200.0, 600.0]
    cur = best_xs.copy()

    for beta in betas:
        if time.time() - t0 > remaining:
            break
        try:
            res = minimize(f, cur, jac=grad_f, method="L-BFGS-B",
                           bounds=[(r0 - 1e-4, 2.0 - r0 + 1e-4)] * N,
                           options={"maxiter": 400, "maxfun": 8000,
                                    "ftol": 1e-12, "gtol": 1e-10})
            if res.x is not None:
                cand = res.x
                v = _row_value(cand, r0)
                if v > best_val + 1e-12:
                    best_val = v
                    best_xs = cand.copy()
                cur = cand
        except Exception:
            pass

    # Assemble the (x, y, r) array.  Thin rectangle: y = r0 for all, r = r0.
    xs = best_xs
    y = np.full(N, r0)
    radii = np.full(N, r0)
    out = np.column_stack((xs, y, radii))
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    arr = construct()
    print("elapsed", time.time() - t0)
    print(arr)
    # self-check
    xs = np.sort(arr[:, 0])
    r = arr[0, 2]
    left = xs[0] - r
    right = 2.0 - xs[-1] - r
    inner = np.diff(xs) - 2.0 * r
    print("left", left, "right", right, "min inner", inner.min())
    print("sum radii", arr[:, 2].sum())
