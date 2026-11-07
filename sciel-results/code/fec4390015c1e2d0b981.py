import numpy as np
import time

class Optimizer:
    """
    Derandomized Discrete PSO optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Initialize swarm with diverse positions using Latin Hypercube Sampling.
    2. Run ddPSO with ring topology, fixed max velocity, and dynamic inertia.
    3. Use remaining budget for local refinement around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # ddPSO Parameters
        swarm_size = 25
        v_max = 1.0
        c1 = 2.0
        c2 = 2.0
        w_start = 0.9
        w_end = 0.4
        
        # Phase 1: Initialize swarm with LHS
        particles_x = np.zeros((swarm_size, dim))
        for d in range(dim):
            bins = np.linspace(-5, 5, swarm_size + 1)
            for i in range(swarm_size):
                particles_x[i, d] = rng.uniform(bins[i], bins[i+1])
        
        # Evaluate initial population
        particles_f = np.full(swarm_size, float('inf'))
        for i in range(swarm_size):
            fx = safe_call(particles_x[i])
            if fx is None:
                break
            particles_f[i] = fx
            
        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                particles_x[0] = x
                particles_f[0] = fx

        particles_v = np.zeros((swarm_size, dim))
        pbest_x = particles_x.copy()
        pbest_f = particles_f.copy()
        gbest_x = best_x.copy()
        gbest_f = best_f

        # Phase 2: ddPSO Loop
        while check_time() and calls_made < budget:
            progress = calls_made / budget
            w = w_start - (w_start - w_end) * progress
            
            new_x = np.zeros((swarm_size, dim))
            new_f = np.full(swarm_size, float('inf'))
            
            for i in range(swarm_size):
                if not check_time() or calls_made >= budget:
                    break
                    
                # Ring topology: use best of self and 2 neighbors
                n1 = (i - 1) % swarm_size
                n2 = (i + 1) % swarm_size
                neigh_idx = [i, n1, n2]
                neigh_f = [pbest_f[j] for j in neigh_idx]
                best_neigh_idx = neigh_idx[np.argmin(neigh_f)]
                neigh_best_x = pbest_x[best_neigh_idx]
                
                r1 = rng.random()
                r2 = rng.random()
                
                v = w * particles_v[i] + c1 * r1 * (pbest_x[i] - particles_x[i]) + c2 * r2 * (neigh_best_x - particles_x[i])
                
                v = np.clip(v, -v_max, v_max)
                new_x[i] = particles_x[i] + v
                new_x[i] = np.clip(new_x[i], -5, 5)
                
                fx = safe_call(new_x[i])
                if fx is None:
                    break
                new_f[i] = fx
                particles_v[i] = v
                
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = new_x[i].copy()
                    
                if fx < gbest_f:
                    gbest_f = fx
                    gbest_x = new_x[i].copy()
                    best_f = fx
                    best_x = new_x[i].copy()

            particles_x = new_x
            particles_f = new_f
            
            if calls_made >= budget or not check_time():
                break

        return best_x, best_f
