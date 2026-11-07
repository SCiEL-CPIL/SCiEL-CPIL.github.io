import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        n_particles = 20
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max = 1.0 # Relative to box size? No, let's use absolute. Box is [-5, 5], width 10.
        # Let's set v_max to 5.0 to allow crossing the whole box in one step if needed.
        
        # Initialize particles
        # Positions: (n_particles, dim)
        p = np.random.uniform(-5, 5, (n_particles, dim))
        # Velocities: (n_particles, dim)
        v = np.random.uniform(-1, 1, (n_particles, dim))
        
        # Evaluate initial population
        f_p = np.array([func(x) for x in p])
        calls_made = n_particles
        
        # Personal bests
        p_best = p.copy()
        f_p_best = f_p.copy()
        
        # Global best
        idx_global = np.argmin(f_p_best)
        g_best = p_best[idx_global].copy()
        f_g_best = f_p_best[idx_global]
        
        # Iterations
        n_iters = budget // n_particles
        if n_iters < 1:
            n_iters = 1
            # If budget < n_particles, we just use what we have
            # But budget is 500, n_particles 20, so 25 iters.
            
        # We might have leftover calls
        remaining = budget - (n_iters * n_particles)
        
        for i in range(n_iters):
            # Update inertia weight
            w = w_start + (w_end - w_start) * (i / n_iters)
            
            # Update velocities and positions
            for j in range(n_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                v[j] = w * v[j] + c1 * r1 * (p_best[j] - p[j]) + c2 * r2 * (g_best - p[j])
                
                # Clip velocity
                v[j] = np.clip(v[j], -5, 5)
                
                p[j] = p[j] + v[j]
                
                # Clip position to box
                p[j] = np.clip(p[j], -5, 5)
                
            # Evaluate new positions
            f_p = np.array([func(x) for x in p])
            calls_made += n_particles
            
            # Update personal bests
            improved = f_p < f_p_best
            p_best[improved] = p[improved]
            f_p_best[improved] = f_p[improved]
            
            # Update global best
            idx_global = np.argmin(f_p_best)
            if f_p_best[idx_global] < f_g_best:
                f_g_best = f_p_best[idx_global]
                g_best = p_best[idx_global].copy()
                
        # If we have remaining calls, use them for a local refinement (Nelder-Mead) around g_best
        if remaining > 0:
            # Simple local search: take 'remaining' random steps around g_best with small noise
            # Or just do more PSO iterations with smaller population?
            # Let's do a simple gradient-free local search (e.g., Nelder-Mead simplified or just random perturbations)
            # Since we have few calls, let's just do random perturbations with decreasing step size.
            current_best = g_best.copy()
            current_best_f = f_g_best
            
            # Use remaining calls for local refinement
            # Step size starts small and decreases
            # We can use the remaining calls to do a simple local search
            # For simplicity, let's just do random perturbations
            step_size = 0.5
            calls_left = remaining
            while calls_left > 0:
                # Generate a batch of local candidates
                n_local = min(calls_left, 10)
                candidates = current_best + np.random.normal(0, step_size, (n_local, dim))
                candidates = np.clip(candidates, -5, 5)
                
                f_cands = np.array([func(x) for x in candidates])
                calls_left -= n_local
                calls_made += n_local
                
                idx_min = np.argmin(f_cands)
                if f_cands[idx_min] < current_best_f:
                    current_best = candidates[idx_min].copy()
                    current_best_f = f_cands[idx_min]
                    
                # Decrease step size
                step_size *= 0.8

        return g_best, f_g_best
