"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel multi-scale differential evolution for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count
            if time.time() > deadline:
                return None
            call_count += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(42)

        # Three populations at different scales
        # Large: global exploration (size 20)
        # Medium: basin identification (size 10)
        # Small: fine-tuning (size 5)
        pop_sizes = [20, 10, 5]
        scales = [3.0, 1.0, 0.3]  # initial step sizes for each population

        # Initialize populations
        populations = []
        fitnesses = []

        for i, (size, scale) in enumerate(zip(pop_sizes, scales)):
            pop = rng.uniform(-5, 5, (size, dim))
            pops_f = []
            for j in range(size):
                f = eval_point(pop[j])
                if f is None:
                    return best_x, best_f
                pops_f.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pop[j].copy()
            populations.append(pop)
            fitnesses.append(np.array(pops_f))

        # Archive of top solutions for exploitation
        archive = [(best_f, best_x.copy())]
        max_archive_size = 5

        # Adaptive crossover rates
        crossover_rates = [0.8, 0.7, 0.6]

        # Main loop: iterate over populations round-robin
        total_steps = 0
        max_steps = 1000  # safety cap on iterations

        while call_count < budget - 5 and time.time() < deadline and total_steps < max_steps:
            for pop_idx in range(len(populations)):
                if call_count >= budget - 5 or time.time() > deadline:
                    break

                pop = populations[pop_idx]
                fit = fitnesses[pop_idx]
                size = len(pop)
                scale = scales[pop_idx]
                cr = crossover_rates[pop_idx]

                # Find current best in this population
                best_idx = np.argmin(fit)
                best_pop_f = fit[best_idx]
                best_pop_x = pop[best_idx].copy()

                # Generate offspring for each individual
                new_pop = pop.copy()
                new_fit = fit.copy()
                improvements = 0

                for i in range(size):
                    if call_count >= budget - 5 or time.time() > deadline:
                        break

                    # Differential evolution: pick two distinct random indices
                    r1, r2 = rng.choice(size, size=2, replace=False)
                    while r1 == i or r2 == i:
                        r1, r2 = rng.choice(size, size=2, replace=False)

                    # Create trial vector
                    F_factor = rng.uniform(0.5, 1.5)
                    trial = pop[r1] + F_factor * (pop[r2] - pop[i])

                    # Crossover with current individual
                    cross_mask = rng.random(dim) < cr
                    if not np.any(cross_mask):
                        cross_mask[rng.integers(dim)] = True

                    trial = np.where(cross_mask, trial, pop[i])
                    trial = np.clip(trial, -5, 5)

                    # Evaluate trial
                    f_trial = eval_point(trial)
                    if f_trial is None:
                        break

                    # Selection
                    if f_trial <= fit[i]:
                        new_pop[i] = trial
                        new_fit[i] = f_trial
                        improvements += 1

                        # Update global best
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()

                # Update population
                populations[pop_idx] = new_pop
                fitnesses[pop_idx] = new_fit

                # Adaptive scale: reduce if improving, increase if stagnant
                if improvements > 0:
                    scales[pop_idx] *= 0.9
                    # Occasionally increase crossover rate for exploration
                    if rng.random() < 0.1:
                        crossover_rates[pop_idx] = min(0.95, crossover_rates[pop_idx] + 0.05)
                else:
                    # Stagnation: increase scale and decrease crossover rate
                    scales[pop_idx] *= 1.3
                    crossover_rates[pop_idx] = max(0.3, crossover_rates[pop_idx] - 0.05)

                # Update archive
                if improvements > 0:
                    archive.append((best_pop_f, best_pop_x.copy()))
                    if len(archive) > max_archive_size:
                        archive = archive[-max_archive_size:]

                total_steps += 1

        # Final polish: use archive to do local refinement
        if best_x is not None and call_count < budget - 3 and time.time() < deadline:
            # Try refining from archive entries
            for f_arch, x_arch in archive[-3:]:
                if call_count >= budget - 3 or time.time() > deadline:
                    break

                sigma_polish = 0.1
                for _ in range(5):
                    if call_count >= budget - 3 or time.time() > deadline:
                        break
                    y = np.clip(x_arch + sigma_polish * rng.standard_normal(dim), -5, 5)
                    fy = eval_point(y)
                    if fy is None:
                        break
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        x_arch = y.copy()
                    else:
                        sigma_polish *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
