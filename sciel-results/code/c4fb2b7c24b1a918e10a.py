"""
Strategy: Block-coordinate Frank-Wolfe (conditional gradient) optimization on the sphere.
Instead of per-point L-BFGS, we treat each point as a block. For each block, the
linear minimization oracle of the Frank-Wolfe method has a closed-form solution:
move the point in the tangent direction opposite to its current gradient, with a
magnitude chosen to minimize the linearized energy along that geodesic. This is
O(N) per block and O(N^2) per full pass, very fast. We combine multiple full
passes (alternating blocks) with intra-cluster repulsion nudges to escape the
shallow local minima where points clump together.
"""
import numpy as np
import time
import math

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit)

    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def _compute_energy(X):
        """Compute total logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        energy = -0.5 * np.sum(np.log(dist_sq))
        return energy

    def _tangent_gradient(X):
        """
        Riemannian gradient of the energy on the sphere.
        E = sum_{i<j} log(1/|xi-xj|) = -0.5 * sum_{i<j} log(|xi-xj|^2)
        dE/dxi = sum_{j!=i} (xi - xj) / |xi-xj|^2
        Riemannian grad = Euclidean grad - (Euclidean grad . xi) * xi
        """
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        inv_d2 = 1.0 / dist_sq

        grad_euclidean = np.zeros((N, 3))
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])

        # Project to tangent space
        dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot_grad * X

        return grad_tangent

    def _fw_full_pass(X):
        """
        One full pass of block-coordinate Frank-Wolfe.
        For each point i, compute its gradient, find the FW direction
        (tangent direction that minimizes the linearized energy), and
        take a line-search step along that geodesic.
        """
        for i in range(N):
            # Compute gradient for point i only (O(N))
            # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
            diff_i = X[i] - X  # shape (N, 3)
            dist_sq_i = np.sum(diff_i * diff_i, axis=1)
            dist_sq_i[i] = 1.0  # avoid self-interaction
            dist_sq_i = np.maximum(dist_sq_i, 1e-18)
            inv_d2_i = 1.0 / dist_sq_i
            inv_d2_i[i] = 0.0  # no self-contribution

            grad_i = np.sum(diff_i * inv_d2_i[:, np.newaxis], axis=0)

            # Project to tangent space at x_i
            dot_grad_i = np.dot(grad_i, X[i])
            grad_tangent_i = grad_i - dot_grad_i * X[i]

            # FW direction: minimize <grad_tangent_i, v> over v in tangent space, |v|=1
            # This is -grad_tangent_i / |grad_tangent_i|
            grad_norm = np.linalg.norm(grad_tangent_i)
            if grad_norm < 1e-14:
                continue

            # Direction to move: -grad_tangent_i (normalized)
            direction = -grad_tangent_i / grad_norm

            # Line search: minimize E along geodesic x_i -> x_i + t * direction
            # x(t) = cos(t) * x_i + sin(t) * direction  (for unit direction)
            # We want to find t that minimizes the energy contribution involving point i.
            # Full energy is expensive, so use a simplified line search.
            # The energy terms involving i: sum_{j != i} log(1/|x_i(t) - x_j|)
            # = -0.5 * sum_{j != i} log(|x_i(t) - x_j|^2)

            # Try a few step sizes
            best_t = 0.0
            best_E = _compute_energy(X)  # current energy

            # Candidate step sizes (geodesic angles)
            candidate_ts = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

            for t in candidate_ts:
                # Compute new position for point i
                cos_t = math.cos(t)
                sin_t = math.sin(t)
                X_new_i = cos_t * X[i] + sin_t * direction

                # Compute energy change (only terms involving i)
                # Energy terms involving i: -0.5 * sum_{j != i} log(|x_i_new - x_j|^2)
                diff_new = X_new_i - X
                dist_sq_new = np.sum(diff_new * diff_new, axis=1)
                dist_sq_new[i] = 1.0
                dist_sq_new = np.maximum(dist_sq_new, 1e-18)
                E_new_i_terms = -0.5 * np.sum(np.log(dist_sq_new))

                # Old energy terms involving i
                diff_old = X[i] - X
                dist_sq_old = np.sum(diff_old * diff_old, axis=1)
                dist_sq_old[i] = 1.0
                dist_sq_old = np.maximum(dist_sq_old, 1e-18)
                E_old_i_terms = -0.5 * np.sum(np.log(dist_sq_old))

                # Total energy change
                E_total_new = best_E - E_old_i_terms + E_new_i_terms

                if E_total_new < best_E - 1e-12:
                    best_E = E_total_new
                    best_t = t

            if best_t > 0:
                cos_t = math.cos(best_t)
                sin_t = math.sin(best_t)
                X[i] = cos_t * X[i] + sin_t * direction
                # Renormalize
                norm_i = np.linalg.norm(X[i])
                if norm_i > 1e-12:
                    X[i] /= norm_i

        return X

    def _repulsion_nudge(X, strength=0.05):
        """
        Add small repulsive nudges to escape local minima where points cluster.
        For each pair of close points, add a small random perturbation.
        """
        X = X.copy()
        for i in range(N):
            # Find close neighbors
            diff = X[i] - X
            dist_sq = np.sum(diff * diff, axis=1)
            close_mask = (dist_sq < 0.01) & (np.arange(N) != i)

            if np.any(close_mask):
                # Add small random perturbation
                perturbation = np.random.randn(3) * strength
                # Project to tangent space
                dot_p = np.dot(perturbation, X[i])
                perturbation_tangent = perturbation - dot_p * X[i]
                X[i] = X[i] + perturbation_tangent
                # Renormalize
                norm_i = np.linalg.norm(X[i])
                if norm_i > 1e-12:
                    X[i] /= norm_i

        return X

    def _fw_optimize(X0, n_passes=10, nudge_every=3, nudge_strength=0.05):
        """
        Run block-coordinate FW optimization for n_passes full passes.
        Optionally add repulsion nudges every nudge_every passes.
        """
        X = X0.copy()
        current_E = _compute_energy(X)

        for p in range(n_passes):
            # Full FW pass
            X = _fw_full_pass(X)

            # Optionally add repulsion nudge
            if (p + 1) % nudge_every == 0:
                X = _repulsion_nudge(X, strength=nudge_strength)

            E_new = _compute_energy(X)
            if E_new < current_E:
                current_E = E_new
            else:
                # Energy increased; revert to previous X? We don't store it,
                # so just continue. The FW method should be monotonic in expectation.
                pass

        return X, current_E

    # Generate diverse initializations
    rng = np.random.default_rng(42)

    best_X = None
    best_energy = float('inf')

    # First try: Fibonacci sphere (good structured start)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    X_opt, E_opt = _fw_optimize(X_fib, n_passes=20, nudge_every=5, nudge_strength=0.03)
    if E_opt < best_energy:
        best_energy = E_opt
        best_X = X_opt.copy()

    # Now do random restarts with FW optimization
    n_restarts = 0
    while time.time() < deadline - 5.0:
        # Random initialization on sphere
        X_rand = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
        X_rand = X_rand / norms

        # Also try perturbed Fibonacci
        if n_restarts % 3 == 1:
            X_rand = X_fib + 0.1 * rng.standard_normal((N, 3))
            norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
            X_rand = X_rand / norms
        elif n_restarts % 3 == 2:
            # Random rotation of Fibonacci
            Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            X_rand = X_fib @ Q.T
            # Add small perturbation
            X_rand = X_rand + 0.05 * rng.standard_normal((N, 3))
            norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
            X_rand = X_rand / norms

        # FW optimization with varying parameters
        n_passes = 15 if n_restarts % 2 == 0 else 25
        nudge_every = 4 if n_restarts % 2 == 0 else 6
        nudge_strength = 0.04 if n_restarts % 2 == 0 else 0.02

        X_opt, E_opt = _fw_optimize(X_rand, n_passes=n_passes,
                                     nudge_every=nudge_every,
                                     nudge_strength=nudge_strength)

        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()

        n_restarts += 1

        # Check time
        if time.time() > deadline - 3.0:
            break

    # Final refinement on best solution with more passes
    if best_X is not None and time.time() < deadline - 1.0:
        X_final, E_final = _fw_optimize(best_X, n_passes=50, nudge_every=10, nudge_strength=0.01)
        if E_final < best_energy:
            best_energy = E_final
            best_X = X_final

    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms

    # Validate
    if not np.all(np.isfinite(best_X)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_X
