"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def fibonacci_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def energy(pts):
        """Compute log(1/d) = -log(d) energy for all pairs."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        # Avoid log(0)
        dist = np.maximum(dist, 1e-12)
        e = -np.log(dist)
        # Sum over i<j
        iu = np.triu_indices(N, k=1)
        return e[iu[0], iu[1]].sum()

    def gradient(pts):
        """Gradient of energy w.r.t. points (on sphere, projected)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-12)
        # dE/dx_i = sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij^2 * (-1) ... 
        # E = sum_{i<j} -log(d_ij)
        # dE/dx_i = sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij  (for i in pair)
        # Actually: d/dx_i [-log(||x_i - x_j||)] = -(x_i - x_j)/||x_i - x_j||^2
        # Wait: d/dx log(||x||) = x/||x||^2, so d/dx (-log d) = -(x_i-x_j)/d^2
        # Gradient of E w.r.t x_i: sum_{j!=i} -(x_i - x_j)/d_ij^2
        inv_d2 = 1.0 / (dist * dist)
        # inv_d2[i,j] = 1/d_ij^2
        inv_d2 = np.where(np.eye(N, dtype=bool), 0.0, inv_d2)
        grad = np.einsum('ij,ijk->ik', inv_d2, diff)
        # Project onto tangent space: grad_t = grad - (grad . x_i) * x_i
        dot = np.einsum('ij,ij->i', grad, pts)
        grad = grad - dot[:, np.newaxis] * pts
        return grad

    def gradient_descent(pts, max_iter=20000, lr_init=1e-4, lr_decay=0.999, tol=1e-12):
        pts = pts.copy()
        lr = lr_init
        best_pts = pts.copy()
        best_e = energy(pts)
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            g = gradient(pts)
            # Adaptive step
            pts_new = normalize(pts - lr * g)
            e_new = energy(pts_new)
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                lr = min(lr * 1.05, 1e-3)
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
        return best_pts, best_e

    def random_sphere(n, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        return normalize(pts)

    # Start with Fibonacci sphere
    pts = fibonacci_sphere()
    pts = normalize(pts)
    best_pts = pts.copy()
    best_e = energy(pts)

    # Run gradient descent
    pts, e = gradient_descent(pts, max_iter=50000, lr_init=5e-5)
    if e < best_e:
        best_e = e
        best_pts = pts

    # Try a few random restarts with gradient descent
    for seed in [42, 123, 999, 7, 2024]:
        if time.time() > deadline - 10.0:
            break
        pts_r = random_sphere(N, seed=seed)
        pts_r, e_r = gradient_descent(pts_r, max_iter=20000, lr_init=1e-4)
        if e_r < best_e:
            best_e = e_r
            best_pts = pts_r

    # Final polish: very small steps
    if time.time() < deadline - 3.0:
        best_pts, best_e = gradient_descent(best_pts, max_iter=20000, lr_init=1e-6, tol=1e-14)

    # Ensure validity
    best_pts = normalize(best_pts)
    # Check no points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if dist.min() < 1e-9:
        # Nudge apart
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-9:
                    best_pts[j] = normalize(best_pts[j] + 1e-8 * np.random.randn(3))

    return best_pts
# EVOLVE-BLOCK-END
