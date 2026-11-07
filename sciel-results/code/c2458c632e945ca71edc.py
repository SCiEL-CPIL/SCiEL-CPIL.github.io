import numpy as np
import time

class Optimizer:
    """
    ddPSO-based optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration and exploitation using Derandomized Discrete PSO (ddPSO).
       - Ring topology for neighborhood information.
       - Dynamic inertia weight (0.9 to 0.4) to balance exploration/exploitation.
       - Velocity clipping to prevent divergence.
    2. Final refinement using adaptive (1+1)-ES around the best particle.
    
    This approach leverages collective memory of the swarm to navigate the 
    multimodal landscape more efficiently than independent local searches,
    providing steady progress for the AOCC metric.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
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
            if not check_time():
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

        # PSO Parameters
        n_particles = 10
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w_start = 0.9
        w_end = 0.4
        max_vel = 1.0
        
        # Total PSO evaluations planned
        # We reserve ~10% of budget for final refinement
        pso_budget = int(budget * 0.9)
        n_generations = pso_budget // n_particles
        
        # Initialize Swarm
        # Uniform initialization in [-5, 5]
        positions = np.random.uniform(-5, 5, (n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        
        pbests = np.copy(positions)
        pbest_fits = np.full(n_particles, float('inf'))
        
        gbest = None
        gbest_fit = float('inf')
        
        # Initial Evaluation
        for i in range(n_particles):
            if not check_time() or calls_made >= budget:
                break
            fx = safe_call(positions[i])
            if fx is None:
                continue
            pbest_fits[i] = fx
            pbests[i] = positions[i].copy()
            if fx < gbest_fit:
                gbest_fit = fx
                gbest = positions[i].copy()

        # Ring Topology Neighborhood
        # Each particle i interacts with i-1 and i+1 (mod n_particles)
        
        for gen in range(n_generations):
            if not check_time() or calls_made >= budget:
                break
            
            # Update inertia weight
            progress = gen / max(1, n_generations - 1)
            w = w_start - (w_start - w_end) * progress
            
            # Update Velocity and Position
            for i in range(n_particles):
                if not check_time() or calls_made >= budget:
                    break
                
                # Determine neighbors in ring topology
                left = (i - 1) % n_particles
                right = (i + 1) % n_particles
                
                # Find local best (lbest) among neighbors and self
                lbest_idx = i
                lbest_fit = pbest_fits[i]
                
                if pbest_fits[left] < lbest_fit:
                    lbest_fit = pbest_fits[left]
                    lbest_idx = left
                if pbest_fits[right] < lbest_fit:
                    lbest_fit = pbest_fits[right]
                    lbest_idx = right
                    
                # Derandomized PSO (ddPSO)
                # In standard PSO, r1, r2 are random. In ddPSO, we often fix them or use deterministic schedules.
                # However, the prompt says "Derandomized Discrete PSO". 
                # A common ddPSO variant uses fixed coefficients or removes the random scaling of c1, c2.
                # Let's use fixed r1=r2=1.0 for the "derandomized" aspect, or small fixed values.
                # To be safe and effective, we will use standard PSO update but with the specified inertia and velocity bounds.
                # The "Derandomized" part often refers to the discrete nature or fixed coefficients. 
                # Given the continuous domain, we interpret this as using fixed c1, c2 multipliers without random scaling per particle, 
                # OR simply that the topology is fixed. 
                # Let's stick to standard velocity update but with the specified constraints.
                
                r1 = 1.0  # Derandomized: fixed cognitive weight
                r2 = 1.0  # Derandomized: fixed social weight
                
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(lbest - x)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - positions[i]) + c2 * r2 * (pbests[lbest_idx] - positions[i])
                
                # Clip velocity to [-max_vel, max_vel]
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                
                # Update position
                new_pos = positions[i] + velocities[i]
                
                # Clip position to bounds [-5, 5]
                new_pos = np.clip(new_pos, -5, 5)
                
                # Evaluate new position
                fx = safe_call(new_pos)
                if fx is None:
                    break
                
                positions[i] = new_pos
                
                # Update personal best
                if fx < pbest_fits[i]:
                    pbest_fits[i] = fx
                    pbests[i] = new_pos.copy()
                    
                    # Update global best
                    if fx < gbest_fit:
                        gbest_fit = fx
                        gbest = new_pos.copy()

        # Phase 2: Refinement
        # Use remaining budget for local refinement around gbest
        if gbest is not None and check_time() and calls_made < budget:
            x = gbest.copy()
            fx = gbest_fit
            sigma = 0.1
            no_improve = 0
            max_no_improve = 20
            
            # Adaptive (1+1)-ES
            while check_time() and calls_made < budget:
                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1 # Expand step
                else:
                    sigma *= 0.9 # Shrink step
                    no_improve += 1
                
                sigma = max(sigma, 1e-5)
                sigma = min(sigma, 2.0)
                
                if no_improve > max_no_improve:
                    # Restart with larger step or break
                    # For AOCC, we want to keep making small improvements if possible
                    # If stuck, try a few random jumps
                    if no_improve > max_no_improve * 2:
                        break
                    sigma = 0.5
                    no_improve = 0

        return best_x, best_f
