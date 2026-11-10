import numpy as np
import math
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Precompute mask for upper triangle pairs
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_idx_i, pair_idx_j = np.where(mask)
    num_pairs = len(pair_idx_i)

    def energy(X):
        """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
        diff = X[pair_idx_i] - X[pair_idx_j]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist = np.maximum(dist, 1e-15)
        return -np.sum(np.log(dist))

    def gradient(X):
        """Compute gradient of energy w.r.t. Cartesian coordinates."""
        diff = X[pair_idx_i] - X[pair_idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-30)
        inv_dist_sq = 1.0 / dist_sq
        # grad_i += diff * inv_dist_sq, grad_j -= diff * inv_dist_sq
        grad = np.zeros((N, 3))
        np.add.at(grad, pair_idx_i, diff * inv_dist_sq[:, np.newaxis])
        np.add.at(grad, pair_idx_j, -diff * inv_dist_sq[:, np.newaxis])
        return grad

    def project_tangent(X, G):
        """Project gradient onto tangent space of sphere at each point."""
        return G - np.sum(G * X, axis=1, keepdims=True) * X

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points(n, rng):
        pts = rng.normal(size=(n, 3))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    def lbfgs_sphere(X0, max_iter=2000, ftol=1e-14, gtol=1e-10):
        """L-BFGS optimization on the sphere using retraction."""
        X = X0.copy()
        m = 20  # L-BFGS memory
        s_list = []
        y_list = []
        rho_list = []

        f = energy(X)
        g = project_tangent(X, gradient(X))

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Check convergence
            gnorm = np.linalg.norm(g)
            if gnorm < gtol:
                break

            # Two-loop recursion to compute search direction
            r = g.copy()
            alphas = np.zeros(m)
            for k in range(len(s_list) - 1, -1, -1):
                alphas[k] = rho_list[k] * np.dot(y_list[k], r)
                r = r - alphas[k] * s_list[k]

            # Initial step (using inverse Hessian approximation)
            if len(s_list) > 0:
                gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            else:
                gamma = 1.0
            q = gamma * r

            for k in range(len(s_list)):
                beta = rho_list[k] * np.dot(y_list[k], q)
                q = q + (alphas[k] - beta) * s_list[k]

            d = -q
            # Ensure d is in tangent space
            d = project_tangent(X, d)
            dnorm = np.linalg.norm(d)
            if dnorm < 1e-15:
                break

            # Backtracking line search
            alpha = 1.0
            c1 = 1e-4
            for ls in range(30):
                # Retraction: move along tangent direction and project back to sphere
                X_new = X + alpha * d
                X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)

                f_new = energy(X_new)
                if not np.isfinite(f_new):
                    alpha *= 0.5
                    continue

                if f_new <= f + c1 * alpha * np.dot(d, g):
                    break
                alpha *= 0.5
            else:
                break

            # Update L-BFGS history
            X_new = X + alpha * d
            X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
            f_new = energy(X_new)

            if f_new >= f:
                # Line search failed, try smaller step
                alpha *= 0.5
                X_new = X + alpha * d
                X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
                f_new = energy(X_new)
                if f_new >= f:
                    break

            s = alpha * d
            g_new = project_tangent(X_new, gradient(X_new))
            y = g_new - g

            sy = np.dot(s, y)
            if sy > 1e-16:
                rho = 1.0 / sy
                s_list.append(s)
                y_list.append(y)
                rho_list.append(rho)
                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

            X = X_new
            f = f_new
            g = g_new

        return X, f

    def cg_sphere(X0, max_iter=500, tol=1e-12):
        """Riemannian conjugate gradient on the sphere."""
        X = X0.copy()
        g = project_tangent(X, gradient(X))
        p = -g
        f = energy(X)

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            gnorm = np.linalg.norm(g)
            if gnorm < tol:
                break

            # Line search
            alpha = 1.0
            c1 = 1e-4
            for ls in range(30):
                X_new = X + alpha * p
                X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
                f_new = energy(X_new)
                if not np.isfinite(f_new):
                    alpha *= 0.5
                    continue
                if f_new <= f + c1 * alpha * np.dot(p, g):
                    break
                alpha *= 0.5
            else:
                break

            X_new = X + alpha * p
            X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
            f_new = energy(X_new)
            g_new = project_tangent(X_new, gradient(X_new))

            # Fletcher-Reeves update
            beta = np.dot(g_new, g_new) / max(np.dot(g, g), 1e-16)
            p = -g_new + beta * p
            # Ensure p is in tangent space
            p = project_tangent(X_new, p)

            X = X_new
            f = f_new
            g = g_new

        return X, f

    def polish(X0, max_iter=5000, ftol=1e-15, gtol=1e-12):
        """High-precision polish using L-BFGS with tight tolerances."""
        return lbfgs_sphere(X0, max_iter=max_iter, ftol=ftol, gtol=gtol)

    # Main optimization
    rng = np.random.default_rng(12345)

    best_X = fibonacci_spiral(N)
    best_E = energy(best_X)

    # Multiple restarts with different initializations
    n_restarts = 8
    time_budget_per_restart = (deadline - start_time) / (n_restarts + 1)

    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break

        if restart == 0:
            X_init = fibonacci_spiral(N)
        elif restart == 1:
            X_init = random_points(N, rng)
        elif restart == 2:
            # Perturbed Fibonacci
            X_init = fibonacci_spiral(N)
            noise = rng.normal(0, 0.1, size=(N, 3))
            X_init = X_init + noise
            X_init = X_init / np.linalg.norm(X_init, axis=1, keepdims=True)
        elif restart == 3:
            # Another random
            X_init = random_points(N, rng)
        elif restart == 4:
            # Perturbed Fibonacci with larger noise
            X_init = fibonacci_spiral(N)
            noise = rng.normal(0, 0.3, size=(N, 3))
            X_init = X_init + noise
            X_init = X_init / np.linalg.norm(X_init, axis=1, keepdims=True)
        elif restart == 5:
            # Random with different seed
            X_init = random_points(N, rng)
        elif restart == 6:
            # Structured grid-like
            X_init = fibonacci_spiral(N)
            # Apply random rotation
            from scipy.stats import special_ortho_group
            R = special_ortho_group.rvs(3, random_state=rng)
            X_init = X_init @ R.T
            X_init = X_init / np.linalg.norm(X_init, axis=1, keepdims=True)
        else:
            X_init = random_points(N, rng)

        # First pass: CG for quick convergence
        X_cg, E_cg = cg_sphere(X_init, max_iter=200, tol=1e-10)

        # Second pass: L-BFGS for better convergence
        X_lb, E_lb = lbfgs_sphere(X_cg, max_iter=1000, ftol=1e-13, gtol=1e-10)

        if E_lb < best_E:
            best_E = E_lb
            best_X = X_lb

    # Final polish on the best solution
    if time.time() < deadline - 10.0:
        X_polished, E_polished = polish(best_X, max_iter=3000, ftol=1e-15, gtol=1e-12)
        if E_polished < best_E:
            best_E = E_polished
            best_X = X_polished

    # Ensure all points are on unit sphere and valid
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)

    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < min_dist:
                min_dist = d

    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        best_X = fibonacci_spiral(N)
        best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)

    return best_X
