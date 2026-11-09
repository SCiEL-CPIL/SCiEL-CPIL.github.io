"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """DFO Optimizer using Particle Swarm Optimization (PSO)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # PSO Parameters
        swarm_size = 10
        c1 = 1.5
        c2 = 1.5
        w = 0.7
        
        lb = -5.0
        ub = 5.0
        
        eval_count = 0
        best_f_global = np.inf
        best_x_global = np.zeros(d)
        
        # Initialize particles
        positions = np.random.uniform(lb, ub, (swarm_size, d))
        velocities = np.random.uniform(-0.5, 0.5, (swarm_size, d))
        
        # Personal bests
        pbest_f = np.full(swarm_size, np.inf)
        pbest_x = np.copy(positions)
        
        # Evaluate initial positions
        for i in range(swarm_size):
            if eval_count >= budget:
                break
            x_clipped = np.clip(positions[i], lb, ub)
            fx = func(x_clipped)
            eval_count += 1
            
            if fx < pbest_f[i]:
                pbest_f[i] = fx
                pbest_x[i] = x_clipped
                
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x_clipped.copy()
                
        # Main loop
        max_iters = (budget - eval_count) // swarm_size
        
        for t in range(max_iters):
            for i in range(swarm_size):
                if eval_count >= budget:
                    break
                    
                # Update velocity
                r1 = np.random.random(d)
                r2 = np.random.random(d)
                
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_x[i] - positions[i]) + c2 * r2 * (best_x_global - positions[i])
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Evaluate
                x_clipped = positions[i]
                fx = func(x_clipped)
                eval_count += 1
                
                # Update personal best
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = x_clipped.copy()
                    
                # Update global best
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = x_clipped.copy()
                    
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
