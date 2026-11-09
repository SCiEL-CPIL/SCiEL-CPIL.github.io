"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Covariance-adaptive (1+1)-ES with multi-scale phases for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Coarse search with large isotropic steps ---
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 3.0
        for i in range(1, min(budget, 50)):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            if fy < best_f:
                best_f, best_x = fy, y.copy()

        # --- Phase 2: Covariance-adaptive refinement ---
        # Estimate covariance from accepted moves to stretch along the valley
        x = best_x.copy()
        fx = best_f
        sigma = 0.5

        # Initialize covariance as identity
        C = np.eye(dim)
        # Track history of accepted moves for covariance estimation
        move_history = []
        max_history = 20

        # Step size adaptation parameters
        sigma = 0.5
        c_sigma = 0.01  # learning rate for sigma
        target_success = 0.2  # desired success rate

        for i in range(50, budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Update covariance from move history (every 5 steps)
            if len(move_history) >= 5 and i % 5 == 0:
                # Compute covariance from recent moves
                moves = np.array(move_history[-max_history:])
                if len(moves) >= dim:
                    # Estimate covariance with regularization
                    C_new = np.cov(moves, rowvar=False)
                    # Regularize to ensure positive definiteness
                    C_new += 1e-6 * np.eye(dim)
                    # Blend with old covariance for stability
                    C = 0.7 * C + 0.3 * C_new
                    # Ensure C is symmetric
                    C = (C + C.T) / 2

            # Generate sample from multivariate normal using Cholesky decomposition
            try:
                L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                step = L @ rng.standard_normal(dim)
            except np.linalg.LinAlgError:
                # Fallback to isotropic if Cholesky fails
                step = rng.standard_normal(dim)

            y = np.clip(x + sigma * step, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accepted move: update history and adapt sigma
                move = y - x
                move_history.append(move)
                if len(move_history) > max_history:
                    move_history.pop(0)
                x, fx = y, fy
                # Increase sigma slightly on success
                sigma *= 1.02
            else:
                # Rejected move: decrease sigma
                sigma *= 0.98

            if fy < best_f:
                best_f, best_x = fy, y.copy()

        # --- Phase 3: Very fine local refinement with small isotropic steps ---
        x = best_x.copy()
        fx = best_f
        sigma = 0.05
        for i in range(budget, budget + 100):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.01
            else:
                sigma *= 0.99
            if fy < best_f:
                best_f, best_x = fy, y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
