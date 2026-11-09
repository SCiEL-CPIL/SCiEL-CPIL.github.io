"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-restart for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling to find a good starting point
        # We use a mix of grid points and random points to ensure coverage
        points = []
        # Center
        points.append(np.zeros(d))
        
        # Axis-aligned points
        for i in range(d):
            for val in [-3.0, -1.5, 0.0, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x)
        
        # Some random points
        n_random = 20
        for _ in range(n_random):
            points.append(rng.uniform(-4.5, 4.5, d))

        # Evaluate initial points
        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: CMA-ES optimization
        # We will run CMA-ES using the remaining budget
        # To handle potential local minima, we might restart if stagnation occurs,
        # but given the small budget, a single robust CMA-ES run is often best.
        
        cma = CMAES(best_x, best_fx, d, rng, call_func, time_ok, budget, calls_made)
        cma.run()
        
        # Update global best
        if cma.best_fx < best_fx:
            best_fx = cma.best_fx
            best_x = cma.best_x

        return best_x, best_fx


class CMAES:
    def __init__(self, x0, fx0, d, rng, func, time_ok, budget, calls_made):
        self.x = x0.copy()
        self.fx = fx0
        self.d = d
        self.rng = rng
        self.func = func
        self.time_ok = time_ok
        self.budget = budget
        self.calls_made = calls_made
        
        # CMA-ES parameters
        self.lam = max(4 + int(3 * np.log(d)), 10)  # Population size
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)
        
        # Step size
        self.sigma = 1.0  # Initial step size, will be adapted
        
        # Covariance matrix
        self.C = np.eye(d)
        self.C_sqrt = np.eye(d)
        
        # Cumulation paths
        self.p_sigma = np.zeros(d)
        self.p_c = np.zeros(d)
        
        # Learning rates
        cs = (self.mueff + 2) / (d + self.mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + cs
        self.c_sigma = cs * ds
        self.c_c = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.c_1 = 2 / ((d + 1.3)**2 + self.mueff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((d + 2)**2 + self.mueff))
        
        self.hsig = None
        self.evals = 0

    def run(self):
        while self.time_ok() and self.evals < self.budget:
            # Generate population
            ys = []
            for i in range(self.lam):
                z = self.rng.standard_normal(self.d)
                y = self.x + self.sigma * self.C_sqrt @ z
                ys.append(y)
            
            # Evaluate population
            f_ys = []
            for y in ys:
                if not self.time_ok():
                    break
                fy = self.func(y)
                f_ys.append(fy)
                self.evals += 1
                if fy < self.fx:
                    self.fx = fy
                    self.x = y.copy()
            
            if len(f_ys) < self.lam:
                break
            
            # Sort by fitness
            idx = np.argsort(f_ys)
            ys = [ys[i] for i in idx]
            f_ys = [f_ys[i] for i in idx]
            
            # Update mean
            old_x = self.x.copy()
            self.x += self.sigma * sum(self.weights[i] * (ys[i] - old_x) for i in range(self.mu))
            
            # Update covariance matrix
            y_diff = np.array([ys[i] - old_x for i in range(self.mu)])
            y_diff = y_diff / self.sigma
            
            # Update cumulation paths
            norm_p_sigma = np.linalg.norm(self.p_sigma)
            hs = (norm_p_sigma / np.sqrt(1 - (1 - self.evals / (d + self.lam))**(2 * self.c_sigma)) <
                  (1.4 + 2 / (self.d + 1)) * np.sqrt((self.mueff + 1) / (self.d + 1)))
            
            self.p_sigma = (1 - self.c_sigma) * self.p_sigma + hs * self.c_sigma / self.sigma * (self.x - old_x)
            
            self.p_c = (1 - self.c_c) * self.p_c + hs * self.c_c * self.p_sigma
            
            delta_hs = (1 - hs) * self.c_c * (2 - self.c_c)
            
            self.C = (1 + self.c_1 + self.c_mu - self.c_1 - self.c_mu) * self.C \
                     + self.c_1 * (np.outer(self.p_c, self.p_c) - self.C) \
                     + self.c_mu * sum(self.weights[i] * np.outer(y_diff[i], y_diff[i]) for i in range(self.mu))
            
            # Ensure symmetry
            self.C = (self.C + self.C.T) / 2
            
            # Update sigma
            self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.p_sigma) / self.ect - 1))
            self.sigma = np.clip(self.sigma, 1e-10, 1e10)
            
            # Update C_sqrt (eigen decomposition)
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-12)
            self.C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Stagnation check: if no improvement in last lam*10 evaluations, break
            # (Simplified: just let it run until budget or time)
            
            self.evals += 1

    @property
    def d_sigma(self):
        # Approximation for dsigma, usually 1 + 2*max(0, sqrt((mueff-1)/(d+1))-1) + cs
        cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        return 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.d + 1)) - 1) + cs

    @property
    def ect(self):
        # Expected norm of standard normal
        return 1 - 1 / (4 * self.d) - 1 / (12 * self.d**2)

# EVOLVE-BLOCK-END
