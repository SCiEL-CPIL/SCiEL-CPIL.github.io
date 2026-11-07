"""Rewritten program using Levenberg-Marquardt method for root finding."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses the Levenberg-Marquardt method:
    - Solves (J^T J + lambda * I) dx = -J^T F
    - Adapts lambda to balance between Newton and Gradient Descent steps
    - Robust to poor initial guesses and ill-conditioned Jacobians
    """
    x = np.array(x0, dtype=float)
    Fx = F(x)
    fx_norm = np.linalg.norm(Fx)
    
    if fx_norm <= 1e-8:
        return x

    # LM parameters
    lambda_val = 1e-3  # Initial damping
    lambda_increase = 10.0
    lambda_decrease = 0.1
    max_iter = 100
    eps = 1e-6  # Step size for finite differences

    def compute_jacobian(x, Fx):
        """Compute Jacobian using central finite differences."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fp = F(x + e)
            Fm = F(x - e)
            J[:, j] = (Fp - Fm) / (2 * eps)
        return J

    for it in range(max_iter):
        if fx_norm <= 1e-8:
            return x

        J = compute_jacobian(x, Fx)
        
        # Compute J^T J and J^T F
        JtJ = J.T @ J
        JtF = J.T @ Fx

        # Solve (J^T J + lambda * I) dx = -J^T F
        # We add a small regularization to I to ensure invertibility if JtJ is singular
        # and to handle the LM damping
        I = np.eye(2)
        A = JtJ + lambda_val * I
        b = -JtF

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, increase lambda significantly and retry
            lambda_val *= lambda_increase
            continue

        x_new = x + dx
        F_new = F(x_new)
        fn_new = np.linalg.norm(F_new)

        # Check improvement
        # The LM criterion often compares the actual reduction to the predicted reduction.
        # A simpler heuristic: if fn_new < fx_norm, we are making progress.
        if fn_new < fx_norm:
            # Accept step
            x = x_new
            Fx = F_new
            fx_norm = fn_new
            # Decrease lambda to allow larger steps (more Newton-like)
            lambda_val *= lambda_decrease
            if lambda_val < 1e-12:
                lambda_val = 1e-12
        else:
            # Reject step
            # Increase lambda to make step smaller (more Gradient Descent-like)
            lambda_val *= lambda_increase
            if lambda_val > 1e12:
                # If lambda is too large, we are doing pure gradient descent with tiny steps.
                # Might be stuck. Break to avoid infinite loop.
                break

    return x
# EVOLVE-BLOCK-END
