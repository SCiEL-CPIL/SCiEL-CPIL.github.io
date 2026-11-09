"""
Structural change: Implement Powell's conjugate direction method as the primary optimizer.
This directly models the Bent Cigar's extreme anisotropy by maintaining and updating
a set of linearly independent search vectors that align with the function's principal axes.
Combined with a small Sobol exploration phase for robustness.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for Bent Cigar (f12) using Powell's conjugate direction method:
    Phase 1: Small Sobol exploration to find a good starting point.
    Phase 2: Powell's method - maintains conjugate directions that adapt to anisotropy.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def make_call(x):
            nonlocal calls_used
            if not check_time() or calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            return fx

        best_x = np.zeros(dim)
        best_fx = float('inf')

        # Phase 1: Small Sobol exploration to find a good starting point
        n_sobol = max(10, min(30, budget // 10))
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * 10.0 - 5.0  # Map to [-5, 5]^d

        for i in range(n_sobol):
            if not check_time() or calls_used >= budget:
                break
            x = sobol_points[i]
            fx = make_call(x)
            if fx is None:
                break
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not check_time() or calls_used >= budget:
            return best_x, best_fx

        # Phase 2: Powell's conjugate direction method
        # Initialize directions as standard basis
        directions = np.eye(dim)
        x = best_x.copy()
        fx = best_fx

        max_cycles = 100
        tol = 1e-10

        for cycle in range(max_cycles):
            if not check_time() or calls_used >= budget:
                break

            x_old = x.copy()
            fx_old = fx
            max_delta = 0.0
            max_dir_idx = 0

            # Line search along each direction
            for i in range(dim):
                if not check_time() or calls_used >= budget:
                    break

                dir_vec = directions[i]
                # Find optimal step along this direction
                alpha = self._line_search(func, make_call, x, dir_vec, fx, check_time, calls_used)
                if alpha is None:
                    break

                x_new = x + alpha * dir_vec
                fx_new = fx + alpha * 0  # We'll re-evaluate

                # Actually evaluate at new point
                fx_new = make_call(x_new)
                if fx_new is None:
                    break

                delta = fx - fx_new
                if delta > max_delta:
                    max_delta = delta
                    max_dir_idx = i

                x = x_new
                fx = fx_new

                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            if not check_time() or calls_used >= budget:
                break

            # Check convergence
            if max_delta < tol:
                break

            # Update directions: replace the direction that gave the largest improvement
            # with the direction from x_old to x_new
            new_dir = x - x_old

            # Check if new_dir is linearly independent of the other directions
            # Remove the direction at max_dir_idx and check
            other_dirs = np.vstack([directions[j] for j in range(dim) if j != max_dir_idx])
            # Check if new_dir is in the span of other_dirs
            if np.linalg.matrix_rank(other_dirs) == dim - 1:
                # new_dir should be independent
                # Check if adding new_dir increases rank
                test_matrix = np.vstack([other_dirs, new_dir])
                if np.linalg.matrix_rank(test_matrix) == dim:
                    # Good, replace
                    directions[max_dir_idx] = new_dir / np.linalg.norm(new_dir)
                else:
                    # Not independent, try to keep old direction but maybe adjust
                    pass
            else:
                # Directions are already degenerate, try to fix
                directions[max_dir_idx] = new_dir / (np.linalg.norm(new_dir) + 1e-10)

            # Optional: reorthogonalize directions to maintain numerical stability
            # Simple Gram-Schmidt reorthogonalization
            for i in range(dim):
                for j in range(i):
                    dot = np.dot(directions[i], directions[j])
                    if abs(dot) > 1e-8:
                        directions[i] -= dot * directions[j]
                norm = np.linalg.norm(directions[i])
                if norm > 1e-10:
                    directions[i] /= norm

        return best_x, best_fx

    def _line_search(self, func, make_call, x, direction, fx_current, check_time, calls_used):
        """
        Simple line search along a direction.
        Returns the optimal alpha (can be negative).
        """
        # Initial step size
        alpha = 1.0
        min_alpha = 1e-8
        max_alpha = 10.0

        # Try to find a good step
        best_alpha = 0.0
        best_fx = fx_current

        # Golden section or simple bisection
        # For simplicity, try a few candidate alphas
        candidates = [alpha, alpha * 0.5, alpha * 0.1, -alpha, -alpha * 0.5, -alpha * 0.1,
                      alpha * 2.0, alpha * 0.01, -alpha * 2.0, -alpha * 0.01]

        for a in candidates:
            if not check_time() or calls_used >= 500:
                break

            x_test = x + a * direction
            # Clip to box
            x_test = np.clip(x_test, -5, 5)

            fx_test = make_call(x_test)
            if fx_test is None:
                break

            if fx_test < best_fx:
                best_fx = fx_test
                best_alpha = a

        return best_alpha if best_alpha != 0.0 else 0.01
# EVOLVE-BLOCK-END
