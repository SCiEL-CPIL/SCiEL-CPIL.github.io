"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Covariance-free adaptive search with random rotation updates."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initialize with a random rotation matrix
        # Generate a random orthogonal matrix using QR decomposition
        A = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        R = Q.copy()  # rotation matrix

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        step_history = []  # store (direction, success) for correlation updates
        window_size = 10
        success_count = 0
        total_steps = 0

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate step in rotated space
            z = rng.standard_normal(dim)
            # Scale to unit length in rotated space
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-10:
                z = z / z_norm
            # Apply rotation to get search direction
            d = R @ z
            # Scale by sigma
            step = sigma * d
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            total_steps += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma *= 1.3
                step_history.append((d, True))
                success_count += 1
            else:
                sigma *= 0.85
                step_history.append((d, False))

            # Keep only last window_size steps
            if len(step_history) > window_size:
                step_history.pop(0)

            # Update rotation based on correlation heuristic every 5 steps
            if total_steps % 5 == 0 and len(step_history) >= window_size:
                # Compute average direction of successful steps
                succ_dirs = [d for d, s in step_history if s]
                fail_dirs = [d for d, s in step_history if not s]

                if succ_dirs:
                    avg_succ = np.mean(succ_dirs, axis=0)
                    avg_succ_norm = np.linalg.norm(avg_succ)
                    if avg_succ_norm > 1e-8:
                        avg_succ = avg_succ / avg_succ_norm

                        # Align first axis of rotation with average success direction
                        # Use a simple Givens-like update: blend current first column with avg_succ
                        new_col0 = R[:, 0]
                        blend = 0.3
                        new_col0 = (1 - blend) * new_col0 + blend * avg_succ
                        new_col0_norm = np.linalg.norm(new_col0)
                        if new_col0_norm > 1e-10:
                            new_col0 = new_col0 / new_col0_norm
                            R[:, 0] = new_col0

                            # Re-orthogonalize the rest using modified Gram-Schmidt
                            for j in range(1, dim):
                                col_j = R[:, j]
                                # Remove component along new_col0
                                col_j = col_j - np.dot(col_j, new_col0) * new_col0
                                # Remove components along previous updated columns
                                for k in range(1, j):
                                    col_j = col_j - np.dot(col_j, R[:, k]) * R[:, k]
                                col_j_norm = np.linalg.norm(col_j)
                                if col_j_norm > 1e-10:
                                    col_j = col_j / col_j_norm
                                    R[:, j] = col_j
                                else:
                                    # Generate a random orthogonal vector
                                    rand_vec = rng.standard_normal(dim)
                                    rand_vec = rand_vec - np.dot(rand_vec, new_col0) * new_col0
                                    for k in range(1, j):
                                        rand_vec = rand_vec - np.dot(rand_vec, R[:, k]) * R[:, k]
                                    rand_vec_norm = np.linalg.norm(rand_vec)
                                    if rand_vec_norm > 1e-10:
                                        R[:, j] = rand_vec / rand_vec_norm

        return best_x, best_fx
# EVOLVE-BLOCK-END
