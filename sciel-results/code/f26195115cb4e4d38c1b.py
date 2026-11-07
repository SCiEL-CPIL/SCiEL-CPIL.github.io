"""Cuckoo Search with Lévy-flight exploration and Nelder-Mead refinement for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Cuckoo Search with Lévy-flight for global exploration, Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0
        rng = np.random.default_rng(777)

        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_ok():
            return time.time() < deadline

        def levy_flight(beta=1.5, scale=1.0):
            """Generate a Lévy flight step."""
            mu = rng.normal(0, 1, dim)
            sigma = rng.normal(0, 1, dim)
            # Mantegna's algorithm for Lévy distribution
            beta_ = beta
            c = 1.0
            beta_param = beta_
            # Simplified: use heavy-tailed distribution
            # Use a mixture: mostly small steps, occasionally large
            step = np.zeros(dim)
            for i in range(dim):
                u = rng.normal(0, 1)
                v = rng.normal(0, 1)
                # Lévy flight step magnitude
                step_mag = abs(u) ** (1.0 / beta_param) * scale
                if v > 0:
                    step[i] = step_mag
                else:
                    step[i] = -step_mag
            return step

        # Phase 1: Cuckoo Search with Lévy-flight
        # Population size
        pop_size = 20
        pa = 0.3  # probability of discovering host's egg

        # Initialize population
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        for i in range(pop_size):
            if call_count >= budget or not time_ok():
                break
            f = evaluate(population[i])

        # Track best in population
        pop_best_idx = np.argmin([evaluate.__wrapped__ if hasattr(evaluate, '__wrapped__') else 0 for _ in range(pop_size)]) if False else 0
        # Actually, let's track properly
        pop_fitness = [float('inf')] * pop_size
        # Re-evaluate isn't ideal; let's track during initial eval
        # Let me restructure: keep a list of (x, f)

        # Better approach: maintain population with fitness
        pop_data = []  # list of (x, f)
        for x in population:
            if call_count >= budget or not time_ok():
                break
            f = evaluate(x)
            pop_data.append((x.copy(), f))

        if not pop_data:
            return best_x, best_f

        # Sort by fitness
        pop_data.sort(key=lambda t: t[1])
        pop_best_x = pop_data[0][0]
        pop_best_f = pop_data[0][1]

        # Cuckoo Search iterations
        max_iterations = 50
        stagnation_counter = 0
        patience = 15

        for it in range(max_iterations):
            if call_count >= budget - 50 or not time_ok():
                break

            if len(pop_data) == 0:
                break

            current_best_f = pop_data[0][1]

            new_pop = []
            for i in range(len(pop_data)):
                if call_count >= budget - 50 or not time_ok():
                    break

                x, f = pop_data[i]

                # Lévy flight step
                step = levy_flight(beta=1.5, scale=1.0)
                new_x = x + step
                new_x = np.clip(new_x, -5.0, 5.0)

                if call_count >= budget - 50 or not time_ok():
                    new_pop.append((x, f))
                    continue

                new_f = evaluate(new_x)

                # Greedy selection: keep better of new_x and x
                if new_f < f:
                    new_pop.append((new_x.copy(), new_f))
                else:
                    new_pop.append((x, f))

            # Replace some nests with new random solutions (discovery)
            for i in range(len(new_pop)):
                if call_count >= budget - 50 or not time_ok():
                    break
                if rng.random() < pa:
                    # Replace with a new random solution
                    new_x = rng.uniform(-4.5, 4.5, dim)
                    new_x = np.clip(new_x, -5.0, 5.0)
                    if call_count >= budget - 50 or not time_ok():
                        continue
                    new_f = evaluate(new_x)
                    # Compare with current best in new_pop
                    if new_f < new_pop[i][1]:
                        new_pop[i] = (new_x.copy(), new_f)

            pop_data = new_pop
            pop_data.sort(key=lambda t: t[1])

            if len(pop_data) == 0:
                break

            new_best_f = pop_data[0][1]

            if new_best_f < current_best_f - 1e-10:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= patience:
                # Restart: reset population around current best
                pop_data = [(best_x.copy(), best_f)]
                for _ in range(pop_size - 1):
                    if call_count >= budget - 50 or not time_ok():
                        break
                    new_x = best_x + rng.normal(0, 0.5, dim)
                    new_x = np.clip(new_x, -5.0, 5.0)
                    new_f = evaluate(new_x)
                    pop_data.append((new_x.copy(), new_f))
                pop_data.sort(key=lambda t: t[1])
                stagnation_counter = 0

        # Phase 2: Nelder-Mead local refinement
        if call_count < budget - 10 and time_ok():
            from scipy.optimize import minimize

            def wrapper(x):
                if not time_ok() or call_count >= budget:
                    return float('inf')
                return evaluate(x)

            # Try Nelder-Mead from the best point
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

            # Try from a few other good points
            if call_count < budget - 10 and time_ok():
                for attempt in range(3):
                    if call_count >= budget - 10 or not time_ok():
                        break
                    x0 = best_x + rng.normal(0, 0.1, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    try:
                        res2 = minimize(wrapper, x0, method='Nelder-Mead',
                                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    except Exception:
                        pass

        # Phase 3: Final random sampling if budget remains
        while call_count < budget and time_ok():
            x = best_x + rng.normal(0, 0.05, dim)
            x = np.clip(x, -5.0, 5.0)
            evaluate(x)

        return best_x, best_f

# EVOLVE-BLOCK-END
