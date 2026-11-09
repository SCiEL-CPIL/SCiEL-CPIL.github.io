import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # 1. Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Helper: project onto unit sphere
    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    # Helper: compute energy and gradient
    def energy_and_gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        energy = -math.fsum(np.log(dist[iu]).tolist())
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return energy, grad

    # Helper: compute log energy only
    def log_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, k=1)
        return -math.fsum(np.log(dist[iu]).tolist())

    # 2. Objective function with gradient for L-BFGS-B
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = project_to_sphere(x)
        e, g = energy_and_gradient(x)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * x, axis=1, keepdims=True) * x
        return e, g_tangent.flatten()

    # 3. L-BFGS-B optimization
    best_x = x_init.copy()
    best_energy = log_energy(best_x)

    # First pass: coarse optimization
    try:
        res = minimize(
            objective_and_grad,
            best_x.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
        )
        if time.time() > deadline:
            pass
        else:
            opt_pts = project_to_sphere(res.x.reshape(N, 3))
            opt_energy = log_energy(opt_pts)
            if opt_energy < best_energy:
                best_x = opt_pts
                best_energy = opt_energy
    except Exception:
        pass

    # Second pass: fine-tune with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            if time.time() < deadline:
                opt_pts2 = project_to_sphere(res2.x.reshape(N, 3))
                opt_energy2 = log_energy(opt_pts2)
                if opt_energy2 < best_energy:
                    best_x = opt_pts2
                    best_energy = opt_energy2
        except Exception:
            pass

    # 4. Newton polish with adaptive step size and line search
    if time.time() < deadline - 5.0:
        pts = best_x.copy()
        current_energy = best_energy
        best_pts = pts.copy()
        best_polish_energy = current_energy

        for it in range(200):
            if time.time() > deadline:
                break

            e, grad = energy_and_gradient(pts)
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            grad_norm = np.linalg.norm(grad_tangent)

            if grad_norm < 1e-14:
                break

            # Adaptive step size
            step = 1.0 / (1.0 + grad_norm * 50)

            # Line search: backtracking with Armijo condition
            alpha = 1.0
            new_pts = None
            new_energy = None
            for _ in range(30):
                candidate = pts - alpha * step * grad_tangent
                candidate = project_to_sphere(candidate)
                cand_energy = log_energy(candidate)

                if cand_energy < current_energy - 1e-4 * alpha * grad_norm * grad_norm:
                    new_pts = candidate
                    new_energy = cand_energy
                    break
                alpha *= 0.5
                if alpha < 1e-8:
                    break

            if new_pts is not None and new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                if current_energy < best_polish_energy:
                    best_polish_energy = current_energy
                    best_pts = pts.copy()
            else:
                # Try smaller fixed step
                small_step = step * 0.01
                candidate = pts - small_step * grad_tangent
                candidate = project_to_sphere(candidate)
                cand_energy = log_energy(candidate)
                if cand_energy < current_energy - 1e-15:
                    pts = candidate
                    current_energy = cand_energy
                    if current_energy < best_polish_energy:
                        best_polish_energy = current_energy
                        best_pts = pts.copy()
                else:
                    break

        if best_polish_energy < best_energy:
            best_x = best_pts
            best_energy = best_polish_energy

    # 5. Final normalization
    x_final = best_x
    norms = np.linalg.norm(x_final, axis=1)
    x_final = x_final / norms[:, np.newaxis]

    return x_final
