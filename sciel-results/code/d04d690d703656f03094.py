import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        # Differential Evolution parameters
        pop_size = 10  # population size
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        polish_budget = 50  # reserve for local polish

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        total_calls += pop_size

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE main loop
        while total_calls < budget - polish_budget and time_ok():
            # For each individual, generate a trial vector
            for i in range(pop_size):
                if not time_ok() or total_calls >= budget - polish_budget:
                    break

                # Select three distinct random individuals
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != b and a != c and b != c and a != i:
                        break

                # Generate trial vector
                v = pop[a] + F * (pop[b] - pop[c])
                # Clip to bounds
                v = np.clip(v, -5, 5)

                # Binary crossover
                r = rng.random(dim)
                j_rand = rng.integers(0, dim)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]

                # Evaluate
                f_trial = func(trial)
                total_calls += 1

                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        # Local polish with L-BFGS-B from best point
        if time_ok() and total_calls < budget:
            def wrapper(x):
                nonlocal total_calls
                if total_calls >= budget:
                    return best_f
                x = np.clip(x, -5, 5)
                f = func(x)
                total_calls += 1
                return f

            try:
                result = minimize(wrapper, best_x, method='L-BFGS-B', 
                                  bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 50, 'maxls': 10})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
