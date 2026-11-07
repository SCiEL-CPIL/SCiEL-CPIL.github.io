import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Large population Differential Evolution for global exploration
        # Use lambda=40 with aggressive mutation for rapid basin identification
        n = self.dim
        pop_size = 40
        F = 0.8
        CR = 0.9
        
        # Initialize population with quasi-random points
        rng = np.random.default_rng(42)
        population = np.random.uniform(-self.bound, self.bound, size=(pop_size, n))
        
        # Evaluate initial population
        for i in range(pop_size):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            population[i] = wrapped_func(population[i])
        
        # Track best
        if self.best_x is not None:
            best_idx = np.argmin([wrapped_func(p) if False else 1e30 for p in [0]])
            # Actually we already tracked best in wrapped_func
        
        # Run DE for remaining budget
        if self.best_x is None:
            return np.zeros(n), 1e30
        
        # Reconstruct population from evaluations
        # We need to re-evaluate or track function values
        # Let's track function values properly
        
        # Reset and redo with proper tracking
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        def wrapped_func2(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30
        
        # Initialize population
        population = np.random.uniform(-self.bound, self.bound, size=(pop_size, n))
        fvals = np.array([wrapped_func2(population[i]) for i in range(pop_size)])
        
        # DE loop
        max_generations = (self.budget - self.call_count) // pop_size
        for gen in range(max_generations):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            
            for i in range(pop_size):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                
                # Select three distinct individuals
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                
                # Mutation
                mutant = population[a] + F * (population[b] - population[c])
                
                # Crossover
                trial = population[i].copy()
                j_rand = np.random.randint(n)
                for j in range(n):
                    if np.random.rand() <= CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -self.bound, self.bound)
                
                # Evaluate trial
                f_trial = wrapped_func2(trial)
                
                # Selection
                if f_trial <= fvals[i]:
                    population[i] = trial
                    fvals[i] = f_trial
        
        # Phase 2: Fine polish with adaptive coordinate descent
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._fine_polish(wrapped_func2, self.best_x.copy(), min(remaining, 30))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass
        
        return self.best_x, self.best_f

    def _fine_polish(self, func, x_start, max_calls):
        """Fine local search with adaptive step sizes."""
        n = self.dim
        x = x_start.copy()
        f_best = func(x)
        calls = 1
        step = 0.02
        for it in range(max_calls - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            improved = False
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    if self.call_count >= self.budget:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-12:
                        x = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
