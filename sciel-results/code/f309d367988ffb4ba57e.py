"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Differential Evolution for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_made = 0

        # Number of restarts and how to split budget
        n_restarts = 5
        # Each restart gets roughly budget/n_restarts calls
        # But we'll do it adaptively: run each restart until time or its budget slice

        # DE parameters
        popsize = 20  # population size per restart
        F = 0.8       # mutation scale
        CR = 0.5      # crossover rate

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Each restart gets a slice of remaining budget
            remaining = budget - calls_made
            # Give this restart at least popsize calls if possible
            restart_budget = min(remaining, max(popsize, remaining // (n_restarts - restart)))
            if restart_budget < 2:
                break

            # Initialize population: mix of random and around global best
            pop = np.zeros((popsize, dim))
            fvals = np.zeros(popsize)

            # First individual: global best (or random for first restart)
            if restart == 0:
                pop[0] = rng.uniform(-5, 5, dim)
            else:
                # Perturb global best
                pop[0] = np.clip(global_best_x + rng.normal(0, 1.0, dim), -5, 5)

            # Rest: random in box
            for i in range(1, popsize):
                pop[i] = rng.uniform(-5, 5, dim)

            # Evaluate initial population
            for i in range(popsize):
                if calls_made >= budget or time.time() > deadline:
                    break
                fi = func(pop[i])
                calls_made += 1
                fvals[i] = fi
                if fi < global_best_f:
                    global_best_f = fi
                    global_best_x = pop[i].copy()

            # DE main loop for this restart
            while calls_made < budget:
                if time.time() > deadline:
                    break

                # Find best in current population
                best_idx = np.argmin(fvals)
                best_f = fvals[best_idx]
                best_x = pop[best_idx].copy()

                # Update global best
                if best_f < global_best_f:
                    global_best_f = best_f
                    global_best_x = best_x.copy()

                # Generate offspring for each individual
                for i in range(popsize):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Select 3 distinct random indices different from i
                    r1, r2, r3 = i
                    while r1 == i:
                        r1 = rng.integers(0, popsize)
                    while r2 == i or r2 == r1:
                        r2 = rng.integers(0, popsize)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = rng.integers(0, popsize)

                    # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = pop[r1] + F * (pop[r2] - pop[r3])

                    # Crossover: binomial
                    trial = pop[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    # Clip to box
                    trial = np.clip(trial, -5, 5)

                    # Evaluate
                    ftrial = func(trial)
                    calls_made += 1

                    # Selection
                    if ftrial <= fvals[i]:
                        pop[i] = trial
                        fvals[i] = ftrial

                    if ftrial < global_best_f:
                        global_best_f = ftrial
                        global_best_x = trial.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
