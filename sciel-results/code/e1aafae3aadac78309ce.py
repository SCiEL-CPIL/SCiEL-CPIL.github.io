"""Rewritten optimizer for BBOB f24 (Lunacek bi-Rastrigin) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Counters and best
        calls = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Helper: evaluate a point safely
        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: Initial Scan ----
        # Lunacek f24 has two basins. We need to find a good starting point.
        # We evaluate a set of diverse points to initialize CMA-ES.
        candidates = []
        candidates.append(np.zeros(n))
        # Random points
        for _ in range(20):
            candidates.append(rng.uniform(-4.5, 4.5, n))
        
        # Evaluate candidates
        for x in candidates:
            if calls >= budget:
                break
            fx = eval_pt(x)
            record(x, fx)

        if time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: CMA-ES ----
        # Initialize CMA-ES parameters
        # Population size
        lam = 4 + 3 * int(np.log(n))
        if lam < 4: lam = 4
        
        # Initial sigma
        sigma = 1.0
        
        # Mean
        m = best_x.copy()
        
        # Covariance matrix (identity initially)
        C = np.eye(n)
        
        # Eigen-decomposition cache
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
        
        # Learning rates
        cc = (4 + lam / n) / (n + 4 + 2 * lam / n)
        cs = (n + 2) / (n + 4 + 2 * lam / n)
        cmu = 2 / ((n + 1.3)**2 + lam)
        cmu = min(cmu, 2 - (cc + cs))
        
        # Path variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Damping for sigma
        dsigma = 1 + 2 * max(0, (sqrt(n) - 1) / (n + 3))
        
        # We will run CMA-ES until budget or time is exhausted
        # We need to be careful with the number of generations
        # Budget is 500. lam is ~7. So ~70 generations max.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                # Sample from N(0, C)
                # Use eigen-decomposition for efficiency and stability
                # z ~ N(0, I)
                z = rng.standard_normal(n)
                # y = eigvecs * (sqrt(eigvals) * z)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                x = m + sigma * y
                x = np.clip(x, -5.0, 5.0)
                fx = eval_pt(x)
                record(x, fx)
                ys.append((x, fx, y))
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # Sort by fitness
            ys.sort(key=lambda item: item[1])
            
            # Update mean
            # mu = lam // 2 (standard choice)
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.array([i + 1 for i in range(mu)]))
            w = w / w.sum()
            
            m_old = m.copy()
            m_new = sum(w[i] * ys[i][0] for i in range(mu))
            
            # Update covariance
            # delta_m = (m_new - m_old) / sigma
            delta_m = (m_new - m_old) / sigma
            
            # Update path covariance pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * C @ delta_m
            
            # Update path sigma ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * delta_m
            
            # Update sigma
            # chi_n is expected norm of N(0, I)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma *= np.exp((dsigma / (n+2)) * (np.linalg.norm(ps)/chi_n - 1))
            
            # Limit sigma
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-12)
            
            # Update mean
            m = m_new
            
            # Update covariance matrix C
            # C = (1 - cs(2-cs-cmu) - cmu) * C + cmu * sum(w_i * (y_i - y_w)(y_i - y_w)^T)
            # where y_w is the weighted mean of the y vectors
            
            # Calculate weighted mean of y
            y_w = sum(w[i] * ys[i][2] for i in range(mu))
            
            C_new = (1 - cs * (2 - cs - cmu) - cmu) * C
            for i in range(mu):
                diff = ys[i][2] - y_w
                C_new += cmu * w[i] * np.outer(diff, diff)
            
            # Ensure symmetry
            C = (C_new + C_new.T) / 2
            
            # Periodic eigen-decomposition to prevent numerical issues
            # and to update the sampling distribution
            # We do this every few generations or if condition number is high
            # For simplicity and speed, we do it every 5 generations
            if int((calls / lam)) % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
