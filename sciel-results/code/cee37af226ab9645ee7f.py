"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Derandomized Discrete Particle Swarm Optimization (ddPSO) for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        # Population size: 20 particles is a good balance for 5D with 500 budget
        pop_size = 20 
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        v_max = 1.0 # Fixed maximum velocity bound
        
        rng = np.random.default_rng(42)
        
        # Initialize positions uniformly in [-5, 5]
        positions = rng.uniform(-5, 5, (pop_size, dim))
        velocities = np.zeros((pop_size, dim))
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in positions])
        calls_made = pop_size
        
        # Initialize personal bests
        pbest = positions.copy()
        pbest_f = fvals.copy()
        
        # Initialize global best
        gbest_idx = np.argmin(pbest_f)
        gbest = pbest[gbest_idx].copy()
        gbest_f = pbest_f[gbest_idx]
        
        # Ring topology neighbors
        # For particle i, neighbors are (i-1) % pop_size and (i+1) % pop_size
        # In standard PSO, social component uses gbest. 
        # In ddPSO with ring, we often use the best of neighbors as the social attractor.
        # However, the prompt suggests "leveraging the swarm's collective memory of personal and global bests".
        # Let's stick to standard PSO social component (gbest) but maybe use ring for local info?
        # Actually, "Derandomized Discrete" usually implies specific update rules.
        # Let's use a standard PSO update but with discrete velocity selection if we were fully discrete.
        # But here we are in continuous space. The "Derandomized Discrete" might refer to the 
        # deterministic choice of which component to apply if we had discrete states.
        # Given the context "operator 'diff'", let's look at the velocity update:
        # v_new = w * v_old + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        # This is the standard "differential" form.
        
        # Let's implement a hybrid: Standard PSO velocity update, but we will 
        # "discretize" the decision of whether to move by rounding the velocity? 
        # No, that's bad for continuous. 
        # Perhaps "Derandomized Discrete" refers to the update strategy where 
        # we don't use random r1, r2 but fixed coefficients? 
        # "Derandomized" often means removing the random factors r1, r2 and using 1.0 or fixed values.
        # Let's try: v = w*v + c1*(pbest - x) + c2*(gbest - x)
        # This is a deterministic gradient-like step influenced by swarm.
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 1.0:
                break
            
            # Linearly decrease inertia weight
            progress = calls_made / budget
            w = w_start - (w_start - w_end) * progress
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                
                x = positions[i]
                pb = pbest[i]
                
                # Derandomized velocity update:
                # v = w * v + c1 * (pb - x) + c2 * (gbest - x)
                # Note: We omit r1, r2 to make it "derandomized"
                # This makes the movement deterministic given the state.
                
                v = w * velocities[i] + c1 * (pb - x) + c2 * (gbest - x)
                
                # Clip velocity to [-v_max, v_max]
                v = np.clip(v, -v_max, v_max)
                
                # New position
                x_new = x + v
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                calls_made += 1
                
                # Update personal best
                if f_new < pbest_f[i]:
                    pbest_f[i] = f_new
                    pbest[i] = x_new.copy()
                    positions[i] = x_new.copy()
                    velocities[i] = v.copy() # Keep velocity for next step? Or reset? 
                    # In PSO, velocity is updated based on new position. 
                    # Usually we store the velocity that led to the new position.
                    
                    # Update global best
                    if f_new < gbest_f:
                        gbest_f = f_new
                        gbest = x_new.copy()
                else:
                    # If not better, we still move to x_new? 
                    # In standard PSO, we always move to the new position.
                    # If it's worse, we might want to backtrack or just continue.
                    # Standard PSO: x = x + v always.
                    positions[i] = x_new.copy()
                    velocities[i] = v.copy()
                    
                    # Optional: If we are stuck, maybe randomize? 
                    # But "Derandomized" suggests we don't.
        
        return gbest, gbest_f
# EVOLVE-BLOCK-END
