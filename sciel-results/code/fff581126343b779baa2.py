"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Improved Particle Swarm Optimization (PSO) continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Adaptive PSO Parameters
        n_particles = 25
        w_start = 0.95
        w_end = 0.35
        c1_start = 2.0
        c1_end = 1.0
        c2_start = 1.0
        c2_end = 2.0
        v_max_factor = 0.6  # Max velocity as fraction of domain range

        # Domain
        lower = -5.0
        upper = 5.0
        range_val = upper - lower
        v_max = v_max_factor * range_val

        # Initialize particles with stratified sampling for better coverage
        positions = np.zeros((n_particles, n))
        for i in range(n_particles):
            # Mix of uniform and grid-based initialization
            if i < 5:
                # Corner points
                positions[i] = rng.choice([lower, upper], n)
            elif i < 10:
                # Face centers
                positions[i] = rng.uniform(lower, upper, n)
                axis = i % n
                positions[i, axis] = 0.0
            else:
                # Random uniform
                positions[i] = rng.uniform(lower, upper, n)
        
        # Velocities
        velocities = rng.uniform(-v_max, v_max, (n_particles, n))

        # Initialize bests
        p_best_pos = positions.copy()
        p_best_val = np.full(n_particles, np.inf)
        g_best_pos = np.zeros(n)
        g_best_val = np.inf

        calls_made = 0

        # Initial evaluation
        for i in range(n_particles):
            f_val = func(positions[i])
            calls_made += 1
            if f_val < p_best_val[i]:
                p_best_val[i] = f_val
                p_best_pos[i] = positions[i]
            if f_val < g_best_val:
                g_best_val = f_val
                g_best_pos = positions[i].copy()

        if calls_made >= budget:
            return g_best_pos, g_best_val

        # Total iterations we can do
        total_iters_possible = (budget - calls_made) // n_particles
        
        # Track stagnation for adaptive restart
        best_val_history = [g_best_val]
        stagnation_counter = 0
        stagnation_threshold = 15  # iterations without improvement triggers restart

        for t in range(total_iters_possible):
            # Adaptive parameters based on progress
            if total_iters_possible > 1:
                progress = t / (total_iters_possible - 1)
            else:
                progress = 0
            
            # Nonlinear inertia decay
            w = w_start + (w_end - w_start) * (1 - np.cos(np.pi * progress)) / 2
            
            # Cognitive and social coefficients
            c1 = c1_start + (c1_end - c1_start) * progress
            c2 = c2_start + (c2_end - c2_start) * progress
            
            # Adaptive velocity limit (tighten as we converge)
            v_max_current = v_max * (1 - 0.5 * progress)
            
            # Check for stagnation and restart if needed
            if t > 0 and t % 5 == 0:
                if best_val_history[-1] - g_best_val < 1e-10:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                if stagnation_counter >= stagnation_threshold:
                    # Restart some particles around g_best
                    n_restart = max(1, n_particles // 4)
                    for i in range(n_restart):
                        idx = rng.choice(n_particles - n_restart, n_restart, replace=False)
                        for j, idx_j in enumerate(idx):
                            # Gaussian perturbation around global best
                            scale = 0.5 * (1 - progress)
                            positions[idx_j] = g_best_pos + rng.normal(0, scale * range_val / 2, n)
                            positions[idx_j] = np.clip(positions[idx_j], lower, upper)
                            velocities[idx_j] = rng.uniform(-v_max_current, v_max_current, n)
                            p_best_pos[idx_j] = positions[idx_j].copy()
                            p_best_val[idx_j] = np.inf
                    stagnation_counter = 0
            
            best_val_history.append(g_best_val)
            
            for i in range(n_particles):
                # Update velocity with adaptive coefficients
                r1 = rng.random(n)
                r2 = rng.random(n)
                
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (p_best_pos[i] - positions[i]) + 
                                c2 * r2 * (g_best_pos - positions[i]))
                
                # Clip velocity
                velocities[i] = np.clip(velocities[i], -v_max_current, v_max_current)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip position to domain
                positions[i] = np.clip(positions[i], lower, upper)
                
                # Evaluate
                f_val = func(positions[i])
                calls_made += 1
                
                # Update pbest
                if f_val < p_best_val[i]:
                    p_best_val[i] = f_val
                    p_best_pos[i] = positions[i].copy()
                
                # Update gbest
                if f_val < g_best_val:
                    g_best_val = f_val
                    g_best_pos = positions[i].copy()
                    
                # If we hit the budget exactly, we can stop early
                if calls_made >= budget:
                    break
                    
            if calls_made >= budget:
                break

        return g_best_pos, g_best_val
# EVOLVE-BLOCK-END
