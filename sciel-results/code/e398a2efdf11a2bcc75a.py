"""Quadratic-surrogate optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic surrogate + local polishing for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        total_budget = self.budget
        n = self.dim

        calls_used = 0

        def wrapper(x):
            nonlocal calls_used
            if calls_used >= total_budget or time.time() > deadline:
                raise StopIteration()
            calls_used += 1
            try:
                v = float(func(x))
            except Exception:
                v = np.inf
            if not np.isfinite(v):
                v = 1e12
            return v

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # ---------------- Phase 1: build quadratic surrogate ----------------
        # Deterministic point set: origin + 5 axis pairs + 6 spherical shell.
        pts = [np.zeros(n)]
        for i in range(n):
            for s in (-1.0, 1.0):
                v = np.zeros(n)
                v[i] = s * 3.0
                pts.append(v)
        # Spherical shell (spherical 2-design: icosahedron directions) scaled to 3.0
        t = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
            [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
            [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1],
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        for d in ico:
            pts.append(d * 3.0)
        # 1 + 10 + 12 = 23 points (well over the 15 needed for a quadratic in 5D)

        X = []
        Y = []
        for p in pts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            try:
                f = wrapper(p)
            except StopIteration:
                break
            except Exception:
                continue
            if np.isfinite(f):
                X.append(np.array(p, dtype=float))
                Y.append(f)
                update_best(p, f)

        # ---------------- Phase 2: iterative surrogate refinement ----------------
        def fit_quadratic(X, Y):
            """Fit f(x) = c + b.x + 0.5 x^T A x.  Returns (c, b, A)."""
            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)
            m, d = X.shape
            # Design matrix: [1, x, x_i x_j (i<=j)]
            cols = [np.ones(m)]
            cols.append(X)
            quad = np.zeros((m, d * (d + 1) // 2))
            k = 0
            for i in range(d):
                for j in range(i, d + 1):
                    quad[:, k] = X[:, i] * X[:, j]
                    k += 1
            M = np.hstack(cols)
            try:
                coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
            except Exception:
                return None
            c = coef[0]
            b = coef[1:1 + d].copy()
            A = np.zeros((d, d))
            k = 0
            for i in range(d):
                for j in range(i, d + 1):
                    A[i, j] = coef[1 + d + k]
                    if i != j:
                        A[j, i] = coef[1 + d + k]
                    k += 1
            return c, b, A

        def surrogate_min(c, b, A, x0):
            """Predicted minimizer of the quadratic surrogate, projected to box."""
            H = A
            try:
                grad = b + H @ x0
                Hinv = np.linalg.inv(H)
                x = x0 - Hinv @ grad
            except Exception:
                # Fallback: damped Newton / gradient step
                grad = b + H @ x0
                x = x0 - 0.5 * grad
            return np.clip(x, -5.0, 5.0)

        # Iteratively add the surrogate's predicted minimum and refit.
        for _ in range(6):
            if calls_used >= total_budget or time.time() > deadline:
                break
            if len(X) < n + 1:
                break
            fit = fit_quadratic(X, Y)
            if fit is None:
                break
            c, b, A = fit
            # Check that the Hessian is positive-definite-ish for a real minimum.
            try:
                ev = np.linalg.eigvalsh(A)
            except Exception:
                ev = None
            if ev is not None and np.min(ev) < -1e-6:
                # Non-convex model; trust it only if it still points somewhere new.
                pass
            x0 = best_x if best_x is not None else np.zeros(n)
            xq = surrogate_min(c, b, A, x0)
            # Only evaluate if it is meaningfully different from existing points.
            if any(np.linalg.norm(xq - x) < 1e-4 for x in X):
                break
            if calls_used >= total_budget or time.time() > deadline:
                break
            try:
                f = wrapper(xq)
            except StopIteration:
                break
            except Exception:
                break
            if not np.isfinite(f):
                break
            X.append(xq)
            Y.append(f)
            update_best(xq, f)

        # ---------------- Phase 3: local polishing (Nelder-Mead style) ----------------
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            # Simple coordinate-descent + reflection polish using a small simplex.
            x = best_x.copy()
            f = best_f
            step = 0.3
            for _ in range(25):
                if calls_used >= total_budget or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls_used >= total_budget or time.time() > deadline:
                        break
                    for s in (-step, step):
                        if calls_used >= total_budget or time.time() > deadline:
                            break
                        trial = x.copy()
                        trial[i] = np.clip(x[i] + s, -5.0, 5.0)
                        try:
                            ft = wrapper(trial)
                        except StopIteration:
                            break
                        except Exception:
                            continue
                        if np.isfinite(ft) and ft < f:
                            x = trial
                            f = ft
                            update_best(x, f)
                            improved = True
                            break
                if improved:
                    step = max(0.05, step * 0.7)
                else:
                    step = max(0.02, step * 0.5)
                    if step < 0.021:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            if calls_used < total_budget:
                try:
                    best_f = wrapper(best_x)
                except Exception:
                    best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
