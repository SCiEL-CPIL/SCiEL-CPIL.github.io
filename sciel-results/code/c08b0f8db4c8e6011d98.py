"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Covariance-free adaptive search with random rotation matrix updated via
    correlation-based heuristic, followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: Adaptive search with rotation matrix
        # Initialize rotation matrix (identity initially)
        R = np.eye(d)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.5
        # Track recent successful steps for correlation-based rotation update
        success_steps = []
        step_buffer_size = 10
        no_improve_count = 0
        max_no_improve = 50

        calls_made = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate perturbation in rotated space
            delta_rot = sigma * rng.standard_normal(d)
            delta = R @ delta_rot
            y = np.clip(x + delta, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Successful step
                fx = fy
                x = y
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Record the successful step direction in original space
                step = y - x  # This is zero after assignment, so use delta
                # Actually, the step was delta, so record delta
                success_steps.append(delta.copy())
                if len(success_steps) > step_buffer_size:
                    success_steps.pop(0)

                # Increase sigma slightly
                sigma *= 1.1

                # Update rotation matrix based on recent successful steps
                if len(success_steps) >= 5:
                    R = self._update_rotation(R, success_steps, rng)
            else:
                no_improve_count += 1
                # Decrease sigma
                sigma *= 0.9

            # If stuck for too long, restart from best point with fresh rotation
            if no_improve_count > max_no_improve:
                x = best_x.copy()
                fx = best_fx
                R = np.eye(d)
                sigma = max(sigma * 0.5, 0.1)
                success_steps = []
                no_improve_count = 0

        # Phase 2: Nelder-Mead local refinement around best point
        if calls_made < budget - 10:
            x, fx = self._nelder_mead(func, best_x, budget - calls_made, d)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx

    def _update_rotation(self, R, steps, rng):
        """Update rotation matrix to align with recent successful step directions."""
        d = R.shape[0]
        # Compute average direction of recent successful steps
        avg_step = np.mean(steps, axis=0)
        norm = np.linalg.norm(avg_step)
        if norm < 1e-10:
            return R

        avg_step = avg_step / norm

        # Project avg_step into current rotated frame
        avg_step_rot = R.T @ avg_step

        # Blend toward aligning first axis with average direction
        # Simple approach: rotate R so that first column points more toward avg_step
        # Use a small rotation toward the target direction
        target_rot = np.zeros(d)
        target_rot[0] = 1.0  # Align first axis

        # Compute rotation that moves current first column toward avg_step_rot
        col0 = R[:, 0]
        # If col0 is already close to avg_step_rot, do nothing
        dot = np.dot(col0, avg_step_rot)
        if dot > 0.99:
            return R

        # Create a small rotation in the plane spanned by col0 and avg_step_rot
        # Simple: blend the first column
        alpha = 0.3  # Learning rate for rotation update
        new_col0 = (1 - alpha) * col0 + alpha * avg_step_rot
        new_col0 = new_col0 / (np.linalg.norm(new_col0) + 1e-10)

        # Reconstruct R with updated first column, keeping other columns orthogonal
        # This is a simplification; for small updates it works reasonably
        R_new = R.copy()
        R_new[:, 0] = new_col0

        # Re-orthogonalize using Gram-Schmidt
        for j in range(1, d):
            # Subtract projection onto new_col0
            proj = np.dot(R_new[:, j], new_col0) * new_col0
            R_new[:, j] = R_new[:, j] - proj
            # Also subtract projections onto previous updated columns
            for k in range(1, j):
                proj = np.dot(R_new[:, j], R_new[:, k]) * R_new[:, k]
                R_new[:, j] = R_new[:, j] - proj
            norm_j = np.linalg.norm(R_new[:, j])
            if norm_j < 1e-10:
                # Column collapsed, regenerate
                R_new[:, j] = rng.standard_normal(d)
                norm_j = np.linalg.norm(R_new[:, j])
            R_new[:, j] /= norm_j

        # Ensure determinant is +1 (proper rotation)
        if np.linalg.det(R_new) < 0:
            R_new[:, -1] *= -1

        return R_new

    def _nelder_mead(self, func, x0, budget, d):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]

        for i in range(1, d + 1):
            xi = x0.copy()
            if abs(x0[i - 1]) < 4.5:
                xi[i - 1] += 0.5
            else:
                xi[i - 1] -= 0.5
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fvals.append(func(xi))

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_fx = fvals[best_idx]
        calls = d + 1

        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        for _ in range(budget - calls):
            if calls >= budget:
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Update best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_fx:
                best_fx = fvals[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
