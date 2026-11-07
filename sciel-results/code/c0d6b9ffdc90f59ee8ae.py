"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    PSO + (1+1)-ES Hybrid Optimizer for BBOB f15 (Rotated Rastrigin).
    
    Strategy:
    1. Particle Swarm Optimization (PSO) for robust global exploration.
       PSO uses population-based search with social/cognitive memory,
       which is effective for multimodal landscapes like Rastrigin.
    2. Adaptive (1+1)-ES for fine-grained local refinement of the best
       point found by PSO.
    
    This combines the population-based exploration strength of PSO
    with the precise exploitation of (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Particle Swarm Optimization ---
        # PSO with a small swarm for efficient exploration of the 5D space.
        
        n_particles = 10
        # Allocate ~65% of budget to PSO, rest to (1+1)-ES refinement
        pso_budget = int(budget * 0.65)
        
        # Initialize particles
        # Use a mix of deterministic and random initialization
        particles = np.zeros((n_particles, dim))
        # First particle at origin (likely near global min for many rotations)
        particles[0] = np.zeros(dim)
        # Some at symmetric points
        particles[1] = np.full(dim, 3.0)
        particles[2] = np.full(dim, -3.0)
        particles[3] = np.array([3.0, -3.0, 3.0, -3.0, 3.0])[:dim]
        particles[4] = np.array([-3.0, 3.0, -3.0, 3.0, -3.0])[:dim]
        # Rest random
        for i in range(5, n_particles):
            particles[i] = rng.uniform(-4.0, 4.0, dim)
        
        # Evaluate initial positions
        p_fits = np.zeros(n_particles)
        for i in range(n_particles):
            if calls >= pso_budget or time.time() > deadline:
                break
            p_fits[i] = eval_func(particles[i])
        
        # Initialize velocities
        velocities = np.zeros((n_particles, dim))
        # Small initial velocities
        for i in range(n_particles):
            velocities[i] = rng.uniform(-0.5, 0.5, dim)
        
        # Personal bests
        p_best = particles.copy()
        p_best_f = p_fits.copy()
        
        # Global best
        g_best_idx = np.argmin(p_best_f)
        g_best = p_best[g_best_idx].copy()
        g_best_f = p_best_f[g_best_idx]
        
        # PSO parameters
        # Inertia weight: decays from 0.9 to 0.4
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        n_pso_calls = pso_budget - calls
        n_iterations = max(1, n_pso_calls // n_particles)
        
        for it in range(n_iterations):
            if calls >= pso_budget or time.time() > deadline:
                break
            
            # Inertia weight decay
            w = 0.9 - 0.5 * (it / max(1, n_iterations))
            
            for i in range(n_particles):
                if calls >= pso_budget or time.time() > deadline:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (p_best[i] - particles[i]) + 
                               c2 * r2 * (g_best - particles[i]))
                
                # Clip velocity to prevent overshooting
                velocities[i] = np.clip(velocities[i], -2.0, 2.0)
                
                # Update position
                new_pos = particles[i] + velocities[i]
                new_pos = np.clip(new_pos, -5.0, 5.0)
                
                # Evaluate
                f_new = eval_func(new_pos)
                
                # Update personal best
                if f_new < p_best_f[i]:
                    p_best[i] = new_pos.copy()
                    p_best_f[i] = f_new
                    
                    # Update global best
                    if f_new < g_best_f:
                        g_best = new_pos.copy()
                        g_best_f = f_new
                
                particles[i] = new_pos

        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Refine the best point found by PSO using (1+1)-ES.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma for local search
            sigma = 0.3
            window = 10
            recent = [False] * window
            idx = 0
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1
                
                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.15
                    elif sr < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.005, 2.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
