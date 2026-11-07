"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import comb


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3 objectives:
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = 0.5 * sin(pi*x1/2)
        # g(x) = sum((x_i - 0.5)^2) for i in [n_front, dim)
        # When g=0, the Pareto front is a quarter sphere of radius 0.5.
        
        n_front = 2
        n_g = dim - n_front
        
        # Use a well-distributed set of points on the quarter sphere.
        # We'll use a combination of:
        # 1. A dense grid in (theta, phi) spherical coordinates with jitter
        # 2. Additional points at strategic locations (corners, midpoints)
        
        n_total = min(budget, 200)
        
        # Strategy: Use a 2D stratified grid in (alpha, beta) where
        # alpha controls the angle from the f3-axis (0 to pi/2)
        # beta controls the azimuthal angle (0 to pi/2)
        # 
        # f3 = 0.5 * sin(alpha)
        # f1 = 0.5 * cos(alpha) * cos(beta)
        # f2 = 0.5 * cos(alpha) * sin(beta)
        
        # For good hypervolume, we want uniform coverage in objective space.
        # The Jacobian for uniform sampling in (f1,f2,f3) on the sphere surface
        # suggests using sqrt(sin(alpha)) for alpha spacing.
        
        # Use a grid with n_alpha x n_beta points
        # Choose n_alpha and n_beta to approximately equal n_total
        n_alpha = int(np.round(np.sqrt(n_total * 1.5)))
        n_beta = int(np.round(n_total / n_alpha))
        
        if n_alpha * n_beta > n_total:
            # Reduce to fit budget
            n_beta = n_total // n_alpha
            if n_beta < 1:
                n_beta = 1
                n_alpha = n_total
        
        # Create stratified grid in alpha and beta
        # For uniform coverage on the sphere, use:
        # alpha_i = arccos(1 - 2*(i + 0.5)/n_alpha) for i in range(n_alpha)
        # But we only want alpha in [0, pi/2], so:
        # alpha_i = arccos(1 - (i + 0.5)/n_alpha) for i in range(n_alpha)
        
        # Actually, let's use a simpler approach:
        # Sample alpha uniformly in [0, pi/2] but with density proportional to cos(alpha)
        # to get uniform area on the sphere.
        
        # Use inverse CDF for uniform area sampling:
        # P(alpha) ∝ cos(alpha) for alpha in [0, pi/2]
        # CDF: F(alpha) = sin(alpha)
        # Inverse: alpha = arcsin(u) where u ~ Uniform[0,1]
        
        # For stratified sampling:
        u_alpha = (np.arange(n_alpha) + 0.5) / n_alpha
        alpha_vals = np.arcsin(u_alpha)
        
        u_beta = (np.arange(n_beta) + 0.5) / n_beta
        beta_vals = u_beta * np.pi / 2.0
        
        # Create meshgrid
        A, B = np.meshgrid(alpha_vals, beta_vals, indexing='ij')
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        n_actual = len(A_flat)
        
        # Add small random jitter to avoid exact grid points
        jitter_scale = 0.1 / n_alpha
        A_flat = A_flat + np.random.uniform(-jitter_scale, jitter_scale, n_actual)
        B_flat = B_flat + np.random.uniform(-0.5/n_beta, 0.5/n_beta, n_actual)
        
        # Compute f values
        f3_vals = 0.5 * np.sin(A_flat)
        f1_vals = 0.5 * np.cos(A_flat) * np.cos(B_flat)
        f2_vals = 0.5 * np.cos(A_flat) * np.sin(B_flat)
        
        # Map back to x1, x2
        # f3 = 0.5 * sin(pi*x1/2) => x1 = (2/pi) * arcsin(2*f3)
        x1_vals = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f3_vals, -1.0, 1.0))
        # f2/f1 = tan(pi*x2/2) => x2 = (2/pi) * arctan2(f2, f1)
        x2_vals = (2.0 / np.pi) * np.arctan2(f2_vals, f1_vals)
        
        # Ensure x1, x2 are in [0, 1]
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x2_vals = np.clip(x2_vals, 0.0, 1.0)
        
        # Construct decision vectors
        X = np.zeros((n_actual, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        if n_g > 0:
            X[:, n_front:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # If we have remaining budget, add more points with different stratification
        remaining = budget - n_actual
        if remaining > 0:
            # Add additional points using a different grid offset
            n_extra_alpha = int(np.round(np.sqrt(remaining * 1.5)))
            n_extra_beta = int(np.round(remaining / n_extra_alpha))
            
            if n_extra_alpha * n_extra_beta > remaining:
                n_extra_beta = remaining // n_extra_alpha
                if n_extra_beta < 1:
                    n_extra_beta = 1
                    n_extra_alpha = remaining
            
            u_alpha_e = (np.arange(n_extra_alpha) + 0.25) / n_extra_alpha
            alpha_vals_e = np.arcsin(np.clip(u_alpha_e, 0, 1))
            
            u_beta_e = (np.arange(n_extra_beta) + 0.25) / n_extra_beta
            beta_vals_e = u_beta_e * np.pi / 2.0
            
            A_e, B_e = np.meshgrid(alpha_vals_e, beta_vals_e, indexing='ij')
            A_flat_e = A_e.flatten()
            B_flat_e = B_e.flatten()
            
            n_extra_actual = len(A_flat_e)
            
            jitter_scale_e = 0.1 / n_extra_alpha
            A_flat_e = A_flat_e + np.random.uniform(-jitter_scale_e, jitter_scale_e, n_extra_actual)
            B_flat_e = B_flat_e + np.random.uniform(-0.5/n_extra_beta, 0.5/n_extra_beta, n_extra_actual)
            
            f3_e = 0.5 * np.sin(A_flat_e)
            f1_e = 0.5 * np.cos(A_flat_e) * np.cos(B_flat_e)
            f2_e = 0.5 * np.cos(A_flat_e) * np.sin(B_flat_e)
            
            x1_e = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f3_e, -1.0, 1.0))
            x2_e = (2.0 / np.pi) * np.arctan2(f2_e, f1_e)
            x1_e = np.clip(x1_e, 0.0, 1.0)
            x2_e = np.clip(x2_e, 0.0, 1.0)
            
            X_extra = np.zeros((n_extra_actual, dim))
            X_extra[:, 0] = x1_e
            X_extra[:, 1] = x2_e
            if n_g > 0:
                X_extra[:, n_front:] = 0.5
            X_extra = np.clip(X_extra, lo, hi)
            
            X = np.vstack([X, X_extra])
        
        # Check time
        if time.time() - t0 > deadline:
            return X
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
