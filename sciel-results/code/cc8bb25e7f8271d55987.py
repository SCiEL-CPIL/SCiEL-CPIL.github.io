"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simulated Annealing with Rastrigin-aware jumps and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The periodicity is 2*pi. Local minima are spaced by ~6.28.
        # The domain is [-5, 5], width 10. So there are roughly 1.6 periods.
        # A jump of size ~3-4 is often enough to escape a local basin.
        
        # Strategy:
        # 1. Start with a few random points to find a good initial basin (cheap exploration).
        # 2. Run Simulated Annealing (SA) with a high initial temperature to allow large jumps.
        #    The cooling schedule will gradually reduce step size.
        # 3. Once temperature is low, switch to a fine-grained local search (Nelder-Mead or similar)
        #    or just continue SA with very small steps to converge to the minimum.
        
        # Budget allocation:
        # 500 calls total.
        # Phase 1: 20 random starts to pick the best initial point.
        # Phase 2: 480 calls for SA/Refinement.
        
        # Phase 1: Global Exploration
        # Evaluate 10 random points to get a sense of the landscape.
        n_initial = 10
        if self.budget < n_initial + 10:
            # Fallback if budget is tiny (not the case here, but safe)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used = 1
            best_x = x
            best_f = fx
            return best_x, best_f

        candidates_x = []
        candidates_f = []
        for _ in range(n_initial):
            if time.time() > deadline or calls_used >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            candidates_x.append(x)
            candidates_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if not candidates_x:
            return best_x, best_f

        # Pick the best initial point for SA
        idx_best = np.argmin(candidates_f)
        x_curr = candidates_x[idx_best].copy()
        f_curr = candidates_f[idx_best]
        
        # Phase 2: Simulated Annealing
        # We have budget_left calls.
        # SA parameters:
        # T_start: High enough to accept worse solutions and jump local minima.
        # Rastrigin penalty is ~10 per dim for being off-grid.
        # A jump of 3 in one dim changes f by ~10*2 + 10*(1-cos) ~ 20-40.
        # So T_start ~ 50-100 seems reasonable to allow large moves.
        T_start = 100.0
        T_end = 1e-6
        alpha = 0.99  # Cooling rate per step. 
        # We want T to decay from 100 to 1e-6 over N steps.
        # T_end = T_start * alpha^N => N = log(T_end/T_start) / log(alpha)
        # N = log(1e-8) / log(0.99) ~ -18.4 / -0.01 ~ 1840 steps.
        # We only have ~480 steps. So we need a faster cooling or different schedule.
        # Let's use a geometric cooling that fits our budget.
        steps_sa = self.budget - calls_used - 20  # Reserve 20 for final polish
        if steps_sa < 10:
             steps_sa = max(1, self.budget - calls_used)
        
        # Calculate alpha to reach T_end in steps_sa
        if steps_sa > 0:
            alpha = (T_end / T_start) ** (1.0 / steps_sa)
        else:
            alpha = 1.0
            
        T = T_start
        
        # Step size in SA: 
        # Standard SA uses a fixed step size or adaptive. 
        # For Rastrigin, a fixed step size relative to the domain width is okay, 
        # but we want large steps early. 
        # Let's make the step size proportional to T? 
        # Or just use a large fixed step size for the first half, then shrink?
        # Actually, the "jump" capability in SA comes from the acceptance probability, 
        # but the move size is usually fixed. 
        # If move size is small, you can't jump over the 2*pi ridges.
        # So we need LARGE moves early.
        
        # Let's use an adaptive step size that starts large and shrinks.
        # sigma_sa starts at 3.0 (half the domain width) and shrinks.
        sigma_sa = 3.0
        sigma_decay = 0.98 # Shrink sigma slightly each step to allow convergence
        
        for i in range(steps_sa):
            if time.time() > deadline or calls_used >= self.budget:
                break
            
            # Generate neighbor
            # Use a mix of large jumps and small moves?
            # Early on: large jumps. Later: small moves.
            # Let's just scale sigma_sa down.
            
            y = x_curr + sigma_sa * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            # Acceptance criterion
            if fy < f_curr:
                x_curr = y
                f_curr = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # Metropolis criterion
                delta_f = fy - f_curr
                if delta_f < 0: # Should not happen if fy > f_curr, but safe
                    x_curr = y
                    f_curr = fy
                else:
                    # Accept with probability exp(-delta_f / T)
                    if T > 1e-10:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x_curr = y
                            f_curr = fy
                            if fy < best_f:
                                best_f = fy
                                best_x = y.copy()
            
            # Cool down
            T *= alpha
            sigma_sa *= sigma_decay
            
            # If sigma becomes too small, we might be stuck. 
            # But T is also low, so we only accept improvements.
            # This effectively becomes a local search.
            
        # Phase 3: Final Polish
        # Use remaining budget for a very fine local search.
        # Since we are likely close to a minimum, use small random steps.
        remaining = self.budget - calls_used
        if remaining > 0 and best_x is not None:
            # Start from best_x found so far
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Use Nelder-Mead if available? No, let's stick to simple random walk with tiny steps.
            # Or just continue SA with T=0 (greedy) and small sigma.
            sigma_polish = 0.1
            for _ in range(remaining):
                if time.time() > deadline or calls_used >= self.budget:
                    break
                
                y = x_curr + sigma_polish * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Shrink sigma slightly to converge
                sigma_polish *= 0.95
                
                # If no improvement in last 10 steps, restart sigma?
                # For simplicity, just keep shrinking. If stuck, it's stuck.
                # But Rastrigin is smooth, so gradient-like descent via small steps works.

        return best_x, best_f
# EVOLVE-BLOCK-END
