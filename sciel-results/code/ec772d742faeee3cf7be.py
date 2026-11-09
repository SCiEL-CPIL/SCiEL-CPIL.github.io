"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: random sampling for early AOCC, then CMA-ES for precision."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget or time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                fx = float(func(x))
                return fx
            except Exception:
                return np.inf

        # Phase 1: Rapid random sampling to establish early baseline for AOCC
        # Use 30% of budget for exploration
        phase1_budget = max(30, int(budget * 0.3))
        
        for _ in range(phase1_budget):
            if call_count >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, d)
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # Phase 2: CMA-ES implementation for precise local optimization
        # We implement a lightweight CMA-ES here to avoid scipy overhead
        # and to have full control over the budget allocation
        
        if call_count < budget and time.time() < deadline:
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(d)))  # Standard CMA-ES population size
            sigma = 1.0  # Initial step size
            c1 = (4 + ln(d)) / ((d + 4) ** 2 + 4)  # Learning rate for covariance
            cmu = min(2, (pop_size - 2 + 0.5) / (d + 2)) / 2
            c_sigma = (pop_size + 1.7) / (pop_size + d + 2.7)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            
            # State variables
            x_mean = best_x.copy()
            C = np.eye(d)
            C_inv = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            # Precompute Cholesky of C
            try:
                B = np.linalg.cholesky(C)
                D = np.diag(B)
                B_inv = np.linalg.inv(B)
            except:
                B = np.eye(d)
                D = np.ones(d)
                B_inv = np.eye(d)
            
            phase2_calls = 0
            max_phase2_calls = budget - call_count
            
            while call_count < budget and time.time() < deadline and phase2_calls < max_phase2_calls:
                # Generate population
                samples = []
                for i in range(pop_size):
                    z = rng.standard_normal(d)
                    # Transform using B and D
                    x_i = x_mean + sigma * (B @ (D * z))
                    x_i = np.clip(x_i, -5, 5)
                    samples.append(x_i)
                
                # Evaluate population
                f_values = []
                for x_i in samples:
                    f_i = wrapped_func(x_i)
                    f_values.append(f_i)
                    phase2_calls += 1
                    
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    
                    if call_count >= budget or time.time() > deadline:
                        break
                
                if len(f_values) < pop_size:
                    break
                
                # Sort by fitness
                indices = np.argsort(f_values)
                y_sorted = [samples[i] for i in indices]
                f_sorted = [f_values[i] for i in indices]
                
                # Update mean
                w = np.zeros(pop_size)
                w[:min(5, pop_size)] = np.log(np.arange(min(5, pop_size) + 1))
                w /= w.sum()
                
                x_new_mean = np.sum([w[i] * y_sorted[i] for i in range(pop_size)], axis=0)
                
                # Update evolution paths
                # Compute weighted mean of selected directions
                diff = x_new_mean - x_mean
                x_mean = x_new_mean
                
                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ (diff / sigma))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update covariance matrix (simplified - use rank-1 update)
                for i in range(min(5, pop_size)):
                    y_i = (y_sorted[i] - x_mean) / sigma
                    C = (1 - c1 - cmu) * C + c1 * np.outer(diff/sigma, diff/sigma) + cmu * w[i] * np.outer(y_i, y_i)
                
                # Ensure C is positive definite
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.clip(eigenvalues, 1e-10, None)
                    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                except:
                    C = np.eye(d)
                
                # Recompute Cholesky
                try:
                    B = np.linalg.cholesky(C)
                    D = np.diag(B)
                    B_inv = np.linalg.inv(B)
                except:
                    B = np.eye(d)
                    D = np.ones(d)
                    B_inv = np.eye(d)
            
            # Final polish with Nelder-Mead if budget remains
            if call_count < budget and time.time() < deadline:
                remaining = budget - call_count
                if remaining > 5:
                    try:
                        from scipy.optimize import minimize
                        res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                       options={'maxiter': remaining, 'adaptive': True})
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
