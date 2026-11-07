import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Particle Swarm Optimization (PSO)
    followed by a local (1+1)-ES polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 290.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        if self.calls >= budget:
            return self.best_x, self.best_f

        # Phase 1: Particle Swarm Optimization (PSO)
        # PSO is effective for multimodal functions like Lunacek.
        n_particles = 30
        # Reserve budget for polish
        polish_budget = 50
        pso_budget = budget - polish_budget
        
        if pso_budget < n_particles:
            # If budget is too small for PSO, just do random search + polish
            pso_budget = 0
            polish_budget = budget

        # Initialize particles
        # Spread particles across the domain
        X = rng.uniform(-5, 5, size=(n_particles, n))
        V = np.zeros((n_particles, n))
        
        # Evaluate initial positions
        f_values = np.empty(n_particles)
        for i in range(n_particles):
            if self.calls >= pso_budget or time.time() > self.deadline:
                break
            x_i = X[i]
            f_i = func(x_i)
            self.calls += 1
            f_values[i] = f_i
            if f_i < self.best_f:
                self.best_f = f_i
                self.best_x = x_i.copy()
        
        if self.calls >= pso_budget or time.time() > self.deadline:
            if self.calls < budget and time.time() < self.deadline:
                self._polish(func, rng)
            return self.best_x, self.best_f

        # Track personal bests
        P_best = X.copy()
        P_best_f = f_values.copy()
        
        # Global best
        g_best_idx = np.argmin(P_best_f)
        G_best = P_best[g_best_idx].copy()
        G_best_f = P_best_f[g_best_idx]
        
        # PSO Parameters
        w = 0.9  # Inertia weight
        c1 = 1.4  # Cognitive coefficient
        c2 = 1.4  # Social coefficient
        w_min = 0.4
        w_max = 0.9
        max_iter_pso = pso_budget // n_particles
        
        # PSO Loop
        for iter_idx in range(max_iter_pso):
            if self.calls >= pso_budget or time.time() > self.deadline:
                break
            
            # Update inertia weight linearly
            w = w_max - (w_max - w_min) * (iter_idx / max_iter_pso)
            
            for i in range(n_particles):
                if self.calls >= pso_budget or time.time() > self.deadline:
                    break
                
                # Update velocity
                r1 = rng.random(n)
                r2 = rng.random(n)
                V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
                
                # Clip velocity to prevent overshooting
                V[i] = np.clip(V[i], -2.0, 2.0)
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], -5, 5)
                
                # Evaluate
                f_i = func(X[i])
                self.calls += 1
                
                if f_i < P_best_f[i]:
                    P_best_f[i] = f_i
                    P_best[i] = X[i].copy()
                    
                    if f_i < G_best_f:
                        G_best_f = f_i
                        G_best = X[i].copy()
                        self.best_f = f_i
                        self.best_x = X[i].copy()

        # Phase 2: Local Polish
        if self.calls < budget and time.time() < self.deadline:
            self._polish(func, rng)

        return self.best_x, self.best_f

    def _polish(self, func, rng):
        """
        Local polish using (1+1)-ES with adaptive step size.
        This refines the global best found by PSO.
        """
        if self.best_x is None:
            return
        if time.time() > self.deadline:
            return
        
        remaining = self.budget - self.calls
        if remaining < 2:
            return
            
        x = self.best_x.copy()
        f_best = self.best_f
        sigma = 0.1  # Start with small step
        
        # Adaptive step size parameters
        sigma_min = 1e-6
        sigma_max = 1.0
        success_ratio = 0.0
        last_success = 0
        
        while self.calls < self.budget:
            if time.time() > self.deadline:
                break
                
            # Generate candidate
            z = rng.standard_normal(self.dim)
            candidate = x + sigma * z
            candidate = np.clip(candidate, -5, 5)
            
            f_cand = func(candidate)
            self.calls += 1
            
            if f_cand < f_best:
                f_best = f_cand
                self.best_f = f_cand
                self.best_x = candidate.copy()
                x = candidate.copy()
                # Increase sigma on success
                sigma = min(sigma * 1.1, sigma_max)
                last_success = self.calls
            else:
                # Decrease sigma on failure
                sigma = max(sigma * 0.9, sigma_min)
            
            # If stuck for too long, try a larger jump
            if self.calls - last_success > 20:
                sigma = min(sigma * 2.0, sigma_max)
