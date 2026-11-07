"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive rotation-based search for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a random point in the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Random rotation matrix for anisotropic adaptation
        Q = np.eye(dim)

        # Initial step size
        sigma = 3.0

        # Track recent successful steps for rotation adaptation
        recent_successes = []
        success_window = 10

        # Phase 1: Global search with adaptive rotation
        calls_made = 1
        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate perturbation in rotated space
            z = rng.standard_normal(dim)
            y_rot = x + sigma * z
            y = x + Q @ (y_rot - x)
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: adapt
                step = y - x
                recent_successes.append(step)

                # Update rotation: align with recent success directions
                if len(recent_successes) >= success_window:
                    # Keep only last success_window steps
                    recent_successes = recent_successes[-success_window:]
                    # Compute principal direction
                    steps_arr = np.array(recent_successes)
                    # Use SVD to find principal direction
                    try:
                        U, S, Vt = np.linalg.svd(steps_arr, full_matrices=False)
                        # The principal direction is the first row of Vt
                        principal = Vt[0]
                        # Blend with current rotation
                        alpha = 0.3
                        new_Q = Q.copy()
                        # Rotate the first axis toward the principal direction
                        # Simple approach: blend the columns
                        for j in range(dim):
                            new_Q[:, j] = (1 - alpha) * Q[:, j] + alpha * np.outer(principal, np.ones(dim))[j] if False else Q[:, j]
                        # Better: just rotate toward principal direction
                        # Use a simple update: project and adjust
                        Q_new = Q.copy()
                        # Align first column with principal direction
                        Q_new[:, 0] = principal / (np.linalg.norm(principal) + 1e-15)
                        # Orjust blend the whole rotation
                        Q = (1 - alpha) * Q + alpha * Q_new
                        # Re-orthonormalize
                        Q, _ = np.linalg.qr(Q)
                    except:
                        pass

                fx, x = fy, y
                sigma *= 1.2  # Increase step size on success

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            else:
                # Failure: decrease step size
                sigma *= 0.85

            # Reset success window occasionally
            if len(recent_successes) > success_window * 2:
                recent_successes = recent_successes[-success_window:]

        # Phase 2: Local refinement (Nelder-Mead style)
        if time.time() < deadline - 30:
            # Use Nelder-Mead with tight tolerance
            from scipy.optimize import minimize

            def wrapper(x_in):
                return func(np.clip(x_in, -5, 5))

            try:
                result = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 200,
                        'maxfev': 100,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun
            except:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
