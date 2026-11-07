"""Optimizer for BBOB f15 (Rastrigin rotated) with Hamiltonian dynamics + DE polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Hamiltonian dynamics + Differential Evolution for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        call_count = 1

        # Phase 1: Hamiltonian dynamics search (use ~60% of budget)
        ham_budget = int(budget * 0.6)
        
        # Initialize particle
        x = best_x.copy()
        v = np.zeros(dim)  # velocity/momentum
        mass = 1.0
        gamma = 0.95  # damping
        dt = 0.5  # time step
        force_scale = 1.0

        # Adaptive parameters
        no_improve = 0
        max_no_improve = 20

        for i in range(ham_budget - 1):
            if time.time() > deadline - 60:
                break
            if call_count >= budget:
                break

            # Compute gradient via finite differences (central difference)
            grad = np.zeros(dim)
            h = 0.01
            for d in range(dim):
                x_plus = x.copy()
                x_plus[d] = min(5.0, x_plus[d] + h)
                x_minus = x.copy()
                x_minus[d] = max(-5.0, x_minus[d] - h)
                fp = func(x_plus)
                fm = func(x_minus)
                call_count += 2
                actual_h = x_plus[d] - x_minus[d]
                if actual_h > 1e-10:
                    grad[d] = (fp - fm) / actual_h

            # Apply force and update velocity (Hamiltonian dynamics with damping)
            force = -force_scale * grad
            v = gamma * v + dt * force / mass

            # Limit velocity to prevent divergence
            v_norm = np.linalg.norm(v)
            max_v = 2.0
            if v_norm > max_v:
                v = v * (max_v / v_norm)

            # Update position
            x_new = x + dt * v
            x_new = np.clip(x_new, -5, 5)

            # Evaluate new position
            fx_new = func(x_new)
            call_count += 1

            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                no_improve = 0
                # Increase force scale slightly when improving
                force_scale *= 1.1
            else:
                no_improve += 1

            x = x_new

            # Reset if stuck for too long
            if no_improve >= max_no_improve:
                # Random restart in Hamiltonian phase
                x = rng.uniform(-5, 5, dim)
                v = np.zeros(dim)
                force_scale = 1.0
                no_improve = 0
                fx_reset = func(x)
                call_count += 1
                if fx_reset < best_f:
                    best_f = fx_reset
                    best_x = x.copy()

        # Phase 2: Differential Evolution polish (use remaining budget)
        de_budget = budget - call_count
        if de_budget > 10:
            # Initialize DE population around best found
            pop_size = min(20, de_budget // 5)
            pop = np.array([best_x + rng.normal(0, 0.5, dim) for _ in range(pop_size)])
            pop = np.clip(pop, -5, 5)
            pop_f = np.array([func(p) for p in pop])
            call_count += pop_size

            # Ensure best_x is in population
            pop[0] = best_x
            pop_f[0] = best_f

            # DE parameters
            F = 0.8  # scale factor
            CR = 0.9  # crossover rate

            while call_count < budget:
                if time.time() > deadline - 30:
                    break

                for i in range(pop_size):
                    if call_count >= budget:
                        break

                    # Select three distinct individuals
                    idx = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(idx, size=3, replace=False)

                    # Mutation
                    trial = pop[a] + F * (pop[b] - pop[c])
                    trial = np.clip(trial, -5, 5)

                    # Crossover
                    r = rng.random(dim)
                    j_rand = rng.integers(0, dim)
                    for d in range(dim):
                        if r[d] <= CR or d == j_rand:
                            trial[d] = pop[i][d]

                    # Evaluate trial
                    ft = func(trial)
                    call_count += 1

                    # Selection
                    if ft < pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = ft
                        if ft < best_f:
                            best_f = ft
                            best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
