"""PSO approach for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    """Particle Swarm Optimization for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # PSO Parameters
        swarm_size = 20
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w_start = 0.9  # Initial inertia weight
        w_end = 0.4    # Final inertia weight

        # Initialize swarm
        positions = rng.uniform(-5, 5, size=(swarm_size, dim))
        velocities = rng.uniform(-2, 2, size=(swarm_size, dim))
        
        # Evaluate initial positions
        for i in range(swarm_size):
            if evals >= budget or time_exceeded():
                break
            try:
                f_val = func(positions[i])
                evals += 1
            except Exception:
                f_val = np.inf
                evals += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = positions[i].copy()
                pbest_x = positions[i].copy()
                pbest_f = f_val
            else:
                pbest_x = positions[i].copy()
                pbest_f = f_val

        # Store personal bests
        pbest_x = positions.copy()
        pbest_f = np.full(swarm_size, np.inf)
        
        # Re-evaluate to get proper pbests if we broke early, or just use the loop above
        # Actually, let's re-initialize pbests properly in the first pass
        # Let's redo the initialization to be clean
        positions = rng.uniform(-5, 5, size=(swarm_size, dim))
        velocities = rng.uniform(-2, 2, size=(swarm_size, dim))
        pbest_x = positions.copy()
        pbest_f = np.full(swarm_size, np.inf)
        gbest_x = np.zeros(dim)
        gbest_f = np.inf
        evals = 0

        for i in range(swarm_size):
            if evals >= budget or time_exceeded():
                break
            try:
                f_val = func(positions[i])
                evals += 1
            except Exception:
                f_val = np.inf
                evals += 1
            
            if f_val < pbest_f[i]:
                pbest_f[i] = f_val
                pbest_x[i] = positions[i].copy()
            
            if f_val < gbest_f:
                gbest_f = f_val
                gbest_x = positions[i].copy()
        
        if gbest_f < best_f:
            best_f = gbest_f
            best_x = gbest_x.copy()

        # Main PSO Loop
        while evals < budget and not time_exceeded():
            # Calculate inertia weight decay
            # We don't know total iterations, so use a fixed decay or time-based
            # Let's use a simple linear decay based on a hypothetical max iterations
            # Max iterations approx budget / swarm_size = 500 / 20 = 25
            # But we might do fewer if we hit budget.
            # Let's just use a constant w=0.7 or decay based on evals/budget
            progress = evals / budget
            w = w_start + (w_end - w_start) * progress

            for i in range(swarm_size):
                if evals >= budget or time_exceeded():
                    break
                
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Update velocity
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_x[i] - positions[i]) + c2 * r2 * (gbest_x - positions[i])
                
                # Clip velocity to prevent explosion
                velocities[i] = np.clip(velocities[i], -5, 5)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)
                
                # Evaluate
                try:
                    f_val = func(positions[i])
                    evals += 1
                except Exception:
                    f_val = np.inf
                    evals += 1
                
                if f_val < pbest_f[i]:
                    pbest_f[i] = f_val
                    pbest_x[i] = positions[i].copy()
                
                if f_val < gbest_f:
                    gbest_f = f_val
                    gbest_x = positions[i].copy()
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = positions[i].copy()

        return best_x, best_f
