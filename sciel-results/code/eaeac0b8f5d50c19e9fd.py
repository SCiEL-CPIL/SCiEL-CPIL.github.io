"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Global stochastic tunneling optimizer for multimodal f22.
    Maintains a population and uses tunneling moves to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Phase 1: Quasi-random scan to build initial population
        n_scan = min(50, budget // 10)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Build initial population from best scan points
        pop_size = 12
        population = []
        # Add best point and some diverse points
        population.append((best_x.copy(), best_f))
        
        # Add diverse random points around the best
        for i in range(pop_size - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = best_x + rng.standard_normal(dim) * 2.0
            x = np.clip(x, -5, 5)
            fx = eval_and_track(x)
            population.append((x.copy(), fx))
        
        # Sort population by fitness
        population.sort(key=lambda t: t[1])

        # Phase 2: Stochastic Tunneling
        # Each iteration: pick a candidate, attempt a tunneling move
        tunneling_steps = 80  # max tunneling iterations
        base_step = 1.0
        
        for it in range(tunneling_steps):
            if calls >= budget or time.time() > deadline:
                break
            
            # Pick a candidate (weighted toward better solutions but with exploration)
            # Use roulette wheel selection with temperature
            temp = max(0.1, 1.0 - it / tunneling_steps)
            fitnesses = np.array([p[1] for p in population])
            min_f = fitnesses.min()
            # Shift to positive
            shifted = fitnesses - min_f + 1e-8
            # Invert so better (lower) gets higher weight
            weights = 1.0 / (shifted ** (1.0 / temp))
            weights /= weights.sum()
            
            idx = rng.choice(len(population), p=weights)
            x_curr, f_curr = population[idx]
            
            # Tunneling move: try to jump to a new basin
            # Generate a random target direction
            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            
            # Try a few tunneling steps
            tunnel_success = False
            for step_mult in [0.5, 1.0, 2.0, 3.0]:
                if calls >= budget or time.time() > deadline:
                    break
                
                step_size = base_step * step_mult
                x_trial = x_curr + step_size * direction
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = eval_and_track(x_trial)
                
                if f_trial < f_curr:
                    # Improvement - update population
                    population[idx] = (x_trial.copy(), f_trial)
                    tunnel_success = True
                    break
                elif f_trial < best_f + 0.1:
                    # Close to best, accept as exploration
                    population[idx] = (x_trial.copy(), f_trial)
                    tunnel_success = True
                    break
            
            if not tunnel_success:
                # If tunneling failed, try a local refinement around current
                # Small random perturbation
                x_trial = x_curr + rng.standard_normal(dim) * 0.3
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = eval_and_track(x_trial)
                if f_trial < f_curr:
                    population[idx] = (x_trial.copy(), f_trial)
            
            # Update population: keep best, replace worst with new candidate
            population.sort(key=lambda t: t[1])
            
            # Occasionally add a completely new random point to maintain diversity
            if it % 10 == 5 and calls < budget - 50:
                x_new = rng.uniform(-5, 5, dim)
                f_new = eval_and_track(x_new)
                # Replace the worst if new point is better
                if f_new < population[-1][1]:
                    population[-1] = (x_new.copy(), f_new)
                    population.sort(key=lambda t: t[1])

        # Phase 3: Local refinement on the best point
        if calls < budget - 20:
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.5
            for _ in range(20):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = eval_and_track(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
                else:
                    step = max(step * 1.1, 0.1)
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        return best_x, best_f

# EVOLVE-BLOCK-END
