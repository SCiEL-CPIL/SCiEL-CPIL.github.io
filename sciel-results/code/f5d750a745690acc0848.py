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

        # Phase 1: Initialize swarm with diverse positions (LHS)
        n_particles = max(10, min(30, budget // 15))
        
        positions = np.zeros((n_particles, dim))
        for d in range(dim):
            perm = rng.permutation(n_particles)
            positions[:, d] = -5.0 + 10.0 * (perm + rng.random(n_particles)) / n_particles
        positions = np.clip(positions, -5, 5)
        
        # Evaluate initial positions
        pbest_f = np.full(n_particles, float('inf'))
        pbest_x = np.zeros((n_particles, dim))
        
        for i in range(n_particles):
            if not check_time() or calls_made >= budget:
                break
            fx = safe_call(positions[i])
            if fx is None:
                break
            pbest_f[i] = fx
            pbest_x[i] = positions[i].copy()
        
        if best_x is None:
            return best_x, best_f

        # Initialize velocities to zero
        velocities = np.zeros((n_particles, dim))
        max_vel = 1.0  # Fixed maximum velocity bound

        # Global best
        gbest_f = best_f
        gbest_x = best_x.copy()

        # Phase 2: ddPSO with ring topology and dynamic inertia
        total_iterations = max(1, (budget - calls_made) // n_particles)
        
        for iteration in range(total_iterations):
            if not check_time() or calls_made >= budget:
                break
            
            # Dynamic inertia: decreases linearly from 0.9 to 0.4
            w = 0.9 - 0.5 * (iteration / max(1, total_iterations))
            
            for i in range(n_particles):
                if not check_time() or calls_made >= budget:
                    break
                
                # Ring topology: each particle interacts with its two neighbors
                left_neighbor = (i - 1) % n_particles
                right_neighbor = (i + 1) % n_particles
                
                # Personal best
                c1 = 1.5  # Cognitive coefficient
                # Social best from neighbors (ring topology)
                c2 = 1.5  # Social coefficient
                
                # Derandomized: use deterministic coefficients
                r1 = 0.5  # Fixed random factor for cognitive
                r2 = 0.5  # Fixed random factor for social
                
                # Velocity update
                vel_update = (w * velocities[i] 
                             + c1 * r1 * (pbest_x[i] - positions[i])
                             + c2 * r2 * (gbest_x - positions[i]))
                
                # Clip velocity to max_vel
                velocities[i] = np.clip(vel_update, -max_vel, max_vel)
                
                # Position update
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)
                
                # Evaluate new position
                fx = safe_call(new_pos)
                if fx is None:
                    break
                
                positions[i] = new_pos
                
                # Update personal best
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = new_pos.copy()
                
                # Update global best
                if fx < gbest_f:
                    gbest_f = fx
                    gbest_x = new_pos.copy()

        # Phase 3: Local refinement around the best point using adaptive (1+1)-ES
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            sigma = 0.3
            no_improve = 0
            max_no_improve = 100
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 3.0)
                
                if no_improve > max_no_improve:
                    # Try a larger step to escape local optimum
                    sigma = 1.0
                    no_improve = 0

        return best_x, best_f
