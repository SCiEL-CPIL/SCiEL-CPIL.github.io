"""
Optimized program: Multi-start global optimization with analytic gradients
for the Hermite coefficient uncertainty bound problem.
Uses L-BFGS-B for fast local convergence from multiple diverse starting points.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes and weights
    # N=25 is sufficient for degree 48 polynomials (P^4)
    nodes, weights = hermgauss(25)
    sqrt2 = np.sqrt(2.0)
    x_nodes = sqrt2 * nodes

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def hermite_p(x, n):
        """Compute physicists' Hermite polynomial H_n(x)."""
        if n == 0:
            return np.ones_like(x)
        elif n == 4:
            return 16.0 * x**4 - 48.0 * x**2 + 12.0
        elif n == 8:
            return 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
        elif n == 12:
            return (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                    - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
        else:
            raise ValueError(f"Unsupported n={n}")

    # Precompute Hermite polynomials at quadrature nodes
    H0_nodes = np.ones_like(x_nodes)
    H4_nodes = hermite_p(x_nodes, 4)
    H8_nodes = hermite_p(x_nodes, 8)
    H12_nodes = hermite_p(x_nodes, 12)

    # Basis matrix for P and Q
    # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Q = a0*H0 + a4*H4 + a8*H8 - a12*H12
    # Note: a12 is linear in a0, a4, a8.
    # a12 = (1/10395)*a0 - (3/10395)*a4 + (105/10395)*a8
    # Let c = [1, -3, 105] / 10395
    c12 = np.array([1.0, -3.0, 105.0]) / 10395.0

    # Coefficients for P:
    # P = a0*(H0 + c12[0]*H12) + a4*(H4 + c12[1]*H12) + a8*(H8 + c12[2]*H12)
    # Coefficients for Q:
    # Q = a0*(H0 - c12[0]*H12) + a4*(H4 - c12[1]*H12) + a8*(H8 - c12[2]*H12)
    
    B0_P = H0_nodes + c12[0] * H12_nodes
    B4_P = H4_nodes + c12[1] * H12_nodes
    B8_P = H8_nodes + c12[2] * H12_nodes
    
    B0_Q = H0_nodes - c12[0] * H12_nodes
    B4_Q = H4_nodes - c12[1] * H12_nodes
    B8_Q = H8_nodes - c12[2] * H12_nodes

    def compute_values(coeffs):
        a0, a4, a8 = coeffs
        P_vals = a0 * B0_P + a4 * B4_P + a8 * B8_P
        Q_vals = a0 * B0_Q + a4 * B4_Q + a8 * B8_Q
        return P_vals, Q_vals

    def objective_and_grad(coeffs):
        """
        Compute the upper bound on C4 and its gradient.
        Bound = (||P||_4 * ||Q||_4) / (||P||_2 * ||Q||_2)
        """
        P_vals, Q_vals = compute_values(coeffs)
        
        # Norms using Gauss-Hermite quadrature
        # ||g||_2^2 = sqrt(2) * sum(w * P^2)
        norm2_g_sq = sqrt2 * np.sum(weights * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(weights * Q_vals**2)
        
        # ||g||_4^4 = sqrt(2) * sum(w * P^4)
        norm4_g_4 = sqrt2 * np.sum(weights * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(weights * Q_vals**4)
        
        if norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15:
            return 1e10, np.zeros(3)
            
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        
        # Gradient calculation
        # Let F = log(bound) = 0.25*log(norm4_g_4) + 0.25*log(norm4_q_4) - 0.5*log(norm2_g_sq) - 0.5*log(norm2_q_sq)
        # dF/da_i = 0.25 * (4 * sum(w P^3 dP/da_i) / norm4_g_4) + 0.25 * (4 * sum(w Q^3 dQ/da_i) / norm4_q_4) 
        #           - 0.5 * (2 * sum(w P dP/da_i) / norm2_g_sq) - 0.5 * (2 * sum(w Q dQ/da_i) / norm2_q_sq)
        # d(bound)/da_i = bound * dF/da_i
        
        # Derivatives of P and Q with respect to a0, a4, a8
        # dP/da0 = B0_P, dP/da4 = B4_P, dP/da8 = B8_P
        # dQ/da0 = B0_Q, dQ/da4 = B4_Q, dQ/da8 = B8_Q
        
        dP_dcoeffs = np.array([B0_P, B4_P, B8_P]) # Shape (3, N)
        dQ_dcoeffs = np.array([B0_Q, B4_Q, B8_Q]) # Shape (3, N)
        
        # Terms for P
        # sum(w * P^3 * dP/da_i)
        w_P3 = weights * P_vals**3
        grad_P4 = np.sum(w_P3[:, np.newaxis] * dP_dcoeffs, axis=1) * sqrt2
        grad_P2 = np.sum((weights * P_vals)[:, np.newaxis] * dP_dcoeffs, axis=1) * sqrt2
        
        # Terms for Q
        w_Q3 = weights * Q_vals**3
        grad_Q4 = np.sum(w_Q3[:, np.newaxis] * dQ_dcoeffs, axis=1) * sqrt2
        grad_Q2 = np.sum((weights * Q_vals)[:, np.newaxis] * dQ_dcoeffs, axis=1) * sqrt2
        
        # Combine
        # d(norm4_g_4)/da_i = 4 * grad_P4_i
        # d(log(norm4_g_4))/da_i = 4 * grad_P4_i / norm4_g_4
        # d(norm2_g_sq)/da_i = 2 * grad_P2_i
        # d(log(norm2_g_sq))/da_i = 2 * grad_P2_i / norm2_g_sq
        
        d_log_bound = (grad_P4 / norm4_g_4) + (grad_Q4 / norm4_q_4) - (grad_P2 / norm2_g_sq) - (grad_Q2 / norm2_q_sq)
        
        grad_bound = bound * d_log_bound
        
        return bound, grad_bound

    def objective(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    # Initial evaluation
    best_coeffs = seed.copy()
    best_val, _ = objective_and_grad(seed)
    
    # Multi-start strategy
    # Define bounds for random search
    # Based on seed, coefficients are small. Let's explore a wider range.
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    # Generate starting points
    starts = [seed.copy()]
    
    # Add random starts
    rng = np.random.default_rng(42)
    num_random = 50
    for _ in range(num_random):
        x0 = rng.uniform(bounds[0][0], bounds[0][1], size=3)
        # Ensure they are not too close to seed
        starts.append(x0)
        
    # Add structured starts around seed
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for i in range(3):
            pert = np.zeros(3)
            pert[i] = scale * (rng.random() - 0.5) * 2.0
            starts.append(seed + pert)

    # Run L-BFGS-B from each start
    # We will limit the number of evaluations per run to ensure we finish in time
    # L-BFGS-B is fast, so we can allow more iterations.
    
    for x0 in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                objective,
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            pass

    # Final polish with Nelder-Mead if time remains
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
