import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Exploration
        # Sample a diverse set of points to identify promising basins.
        n_init = 30
        candidates = []

        # Always try the origin
        eval_func(np.zeros(dim))

        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Particle Swarm Optimization (PSO)
        # Structural change: Use population-based exploration instead of single-point (1+1)-ES.
        # PSO is effective for multimodal landscapes like Rastrigin.
        
        # Select top K candidates to initialize the swarm
        top_k = min(10, len(candidates))
        swarm_size = 20
        
        # Initialize particles
        particles_x = np.zeros((swarm_size, dim))
        particles_v = np.zeros((swarm_size, dim))
        particles_f = np.zeros(swarm_size)
        
        # Initialize positions around the top candidates
        # Interleave top candidates and random points to ensure diversity
        for i in range(swarm_size):
            if i < top_k:
                particles_x[i] = candidates[i][1]
            else:
                particles_x[i] = rng.uniform(-5, 5, dim)
            
            particles_f[i] = eval_func(particles_x[i])
            
            if particles_f[i] < best_f:
                best_f = particles_f[i]
                best_x = particles_x[i].copy()

        # Personal bests
        pbest_x = particles_x.copy()
        pbest_f = particles_f.copy()
        
        # Global best
        gbest_x = best_x.copy()
        gbest_f = best_f

        # PSO Parameters
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w = 0.9   # Inertia weight
        w_min = 0.4
        w_max = 0.9
        v_max = 2.0  # Velocity clamp

        max_iterations = 100  # Max PSO iterations
        
        for it in range(max_iterations):
            if calls >= budget or time.time() > deadline:
                break
            
            # Update inertia weight (linear decrease)
            w = w_max - (w_max - w_min) * (it / max_iterations)
            
            for i in range(swarm_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                particles_v[i] = w * particles_v[i] + c1 * r1 * (pbest_x[i] - particles_x[i]) + c2 * r2 * (gbest_x - particles_x[i])
                
                # Clamp velocity
                particles_v[i] = np.clip(particles_v[i], -v_max, v_max)
                
                # Update position
                particles_x[i] = particles_x[i] + particles_v[i]
                particles_x[i] = np.clip(particles_x[i], -5, 5)
                
                # Evaluate
                f_i = eval_func(particles_x[i])
                
                # Update personal best
                if f_i < pbest_f[i]:
                    pbest_f[i] = f_i
                    pbest_x[i] = particles_x[i].copy()
                    
                    # Update global best
                    if f_i < gbest_f:
                        gbest_f = f_i
                        gbest_x = particles_x[i].copy()

        # Phase 3: Final Polish
        # If we have budget left, do a very fine search around best_x using (1+1)-ES
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            remaining = budget - calls
            max_polish_steps = min(remaining, 50)
            
            for _ in range(max_polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_f
                    sigma *= 0.9
                else:
                    sigma *= 0.95
                sigma = max(sigma, 0.001)

        return best_x, best_f
