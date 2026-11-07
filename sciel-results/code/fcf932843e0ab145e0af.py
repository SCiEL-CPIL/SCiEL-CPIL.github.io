"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # BO State
        X = []  # List of points
        Y = []  # List of function values
        
        # Hyperparameters for GP
        # Length scales: start with 1.0, bounds [0.1, 5.0]
        # Signal variance: start with 1.0, bounds [0.1, 10.0]
        # Noise variance: small, since we assume deterministic func (or very low noise)
        # We will optimize hyperparameters occasionally or use fixed ones for speed.
        # Given 500 calls and 5D, optimizing hyperparameters every 10-20 steps is reasonable.
        
        # Initial design: 10 points using Halton sequence for low-discrepancy coverage
        def halton_sequence(num_points, dim):
            """Generate Halton sequence points in [0, 1]^dim."""
            points = []
            for i in range(1, num_points + 1):
                point = []
                for base in range(2, 2 + dim):
                    f = 1.0
                    r = 0.0
                    i_copy = i
                    while i_copy > 0:
                        i_copy, rem = divmod(i_copy, base)
                        f /= base
                        r += f * rem
                    point.append(r)
                points.append(point)
            return np.array(points)

        # Map Halton [0,1] to [-5,5]
        halton_points = halton_sequence(10, self.dim)
        X_init = halton_points * 10.0 - 5.0
        
        for x in X_init:
            if self.calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            self.calls_made += 1
            f_val = func(x)
            X.append(x)
            Y.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
        
        if not X:
            # Fallback if something went wrong immediately
            return best_x, best_f

        X = np.array(X)
        Y = np.array(Y)
        
        # GP Matern 5/2 Kernel
        def matern52_kernel(X1, X2, length_scale, sigma):
            """
            Compute Matern 5/2 kernel matrix between X1 (n, d) and X2 (m, d).
            """
            # Compute pairwise distances
            # X1: (n, d), X2: (m, d)
            # dist^2 = ||x1||^2 + ||x2||^2 - 2 x1.x2
            x1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
            x2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
            dist_sq = x1_sq + x2_sq - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            dist = np.sqrt(dist_sq)
            
            # Matern 5/2: (1 + sqrt(3)*r + (3/5)*r^2) * exp(-sqrt(3)*r) / (l^2)
            r = dist / length_scale
            val = (1.0 + np.sqrt(3.0) * r + 0.6 * r**2) * np.exp(-np.sqrt(3.0) * r)
            return sigma**2 * val

        def log_likelihood(theta, X, Y):
            """
            Negative log marginal likelihood for GP.
            theta = [log(length_scale), log(sigma), log(noise)]
            """
            try:
                log_l = theta[0]
                log_sig = theta[1]
                log_noise = theta[2]
                
                l = np.exp(log_l)
                sig = np.exp(log_sig)
                noise = np.exp(log_noise)
                
                n = len(X)
                K = matern52_kernel(X, X, l, sig)
                K += noise * np.eye(n)
                
                # Add jitter for numerical stability
                K += 1e-6 * np.eye(n)
                
                # Compute log marginal likelihood
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e6  # Large penalty for non-positive definite
                
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                
                # -0.5 * (Y' K^-1 Y + log(det(K)) + n log(2pi))
                nll = 0.5 * (Y @ alpha + log_det + n * np.log(2.0 * np.pi))
                return nll
                
            except Exception:
                return 1e6

        def predict(X_new, theta):
            """
            Predict mean and variance at X_new given theta.
            """
            log_l, log_sig, log_noise = theta
            l = np.exp(log_l)
            sig = np.exp(log_sig)
            noise = np.exp(log_noise)
            
            K = matern52_kernel(X, X, l, sig)
            K += noise * np.eye(len(X))
            K += 1e-6 * np.eye(len(X))
            
            Ks = matern52_kernel(X, X_new, l, sig)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                L = np.linalg.cholesky(K + 1e-4 * np.eye(len(X)))
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            
            mu = Ks.T @ alpha
            v = np.linalg.solve(L, Ks)
            var = sig**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-12)
            
            return mu, var

        # Optimization loop
        # We will update hyperparameters every 15 steps to balance cost and adaptation
        # Initial hyperparameters: length_scale=1.0, sigma=1.0, noise=1e-4
        theta = np.array([0.0, 0.0, -9.21]) # log(1.0), log(1.0), log(1e-4)
        
        # Acquisition: Expected Improvement
        def expected_improvement(X_cand, theta, f_best):
            """
            Calculate EI for a batch of candidate points X_cand.
            Returns EI values.
            """
            mu, var = predict(X_cand, theta)
            sigma = np.sqrt(var)
            # Avoid division by zero
            sigma = np.maximum(sigma, 1e-12)
            
            # Standardized improvement
            Z = (f_best - mu) / sigma
            EI = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return EI

        # We will use a simple local search to maximize EI
        # Since we have a budget of 500, and GP prediction is O(N^3) for update, O(N^2) for pred,
        # N is small (max 500). 500^3 is 125M ops, which is fine for numpy in 300s.
        # However, optimizing EI requires many predictions.
        # We can use a coarse grid or random restarts for EI maximization.
        
        # To save time, we will use a small number of random candidates + local polish
        # Or simply use scipy minimize on the negative EI function.
        
        while self.calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Update hyperparameters every 15 steps
            if (self.calls_made - 10) % 15 == 0 and len(X) > 10:
                # Optimize hyperparameters
                # Bounds for log params:
                # l in [0.1, 5] -> log in [-2.3, 1.6]
                # sig in [0.1, 10] -> log in [-2.3, 2.3]
                # noise in [1e-5, 1e-2] -> log in [-11.5, -4.6]
                bounds = [(-2.3, 1.6), (-2.3, 2.3), (-11.5, -4.6)]
                # Use current theta as start
                try:
                    res_hyp = minimize(log_likelihood, theta, args=(X, Y), method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                    if res_hyp.fun < log_likelihood(theta, X, Y):
                        theta = res_hyp.x
                except Exception:
                    pass
            
            # Maximize EI
            # We need to find x that maximizes EI(x)
            # We will use a few random starts and local minimization of -EI
            
            def neg_ei(x):
                if self.calls_made >= self.budget:
                    return 0.0
                x = np.clip(x, -5, 5)
                # Predict at single point
                try:
                    ei = expected_improvement(x.reshape(1, -1), theta, best_f)[0]
                    return -ei
                except Exception:
                    return 0.0
            
            best_x_next = None
            best_neg_ei = float('inf')
            
            # Try 5 random starts for EI maximization
            rng = np.random.default_rng(42 + self.calls_made)
            for _ in range(5):
                x0 = rng.uniform(-5, 5, size=self.dim)
                try:
                    res_ei = minimize(neg_ei, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 0.01, 'fatol': 1e-4})
                    if res_ei.fun < best_neg_ei:
                        best_neg_ei = res_ei.fun
                        best_x_next = res_ei.x
                except Exception:
                    pass
            
            if best_x_next is None:
                # Fallback to random point
                best_x_next = rng.uniform(-5, 5, size=self.dim)
            
            # Evaluate the best candidate
            best_x_next = np.clip(best_x_next, -5, 5)
            self.calls_made += 1
            f_val = func(best_x_next)
            
            X.append(best_x_next)
            Y.append(f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = best_x_next.copy()
                
            # Update X, Y arrays for next iteration
            X = np.array(X)
            Y = np.array(Y)
            
            # Optional: If we have very few calls left, stop optimizing EI to save time
            if self.calls_made > 450:
                # Just do random exploration if GP is too slow?
                # No, GP is fast enough. Just break if time is tight.
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
