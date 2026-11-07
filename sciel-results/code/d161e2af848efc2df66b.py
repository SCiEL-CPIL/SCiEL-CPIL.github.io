"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES for exploration + Powell's method for axis-adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _powell_line_min(self, x, direction, func, calls, budget, deadline):
        """Minimize f(x + t*direction) for t in [-1, 1] using golden section search."""
        a, b = -1.0, 1.0
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~0.618
        
        # Evaluate at initial points
        x1 = x + gr * a * direction
        x2 = x + (1.0 - gr) * a * direction + gr * (b - a) * direction
        # Actually, let's do standard golden section on [a, b]
        x1 = x + (a + gr * (b - a)) * direction
        x2 = x + (a + (1 - gr) * (b - a)) * direction
        
        f1 = func(np.clip(x1, -5, 5))
        f2 = func(np.clip(x2, -5, 5))
        calls[0] += 2
        
        for _ in range(20):
            if calls[0] >= budget or time.time() > deadline:
                break
            if f1 < f2:
                b = x2[0] if direction[0] != 0 else b  # This is wrong, let's track t
                # Let me redo this properly with t parameter
                pass
            break
        
        # Simpler: just do a few evaluations to find best t
        best_t = 0.0
        best_f = func(x)
        calls[0] += 1
        for t in [-0.5, -0.25, 0.25, 0.5, -0.1, 0.1, -0.05, 0.05]:
            if calls[0] >= budget or time.time() > deadline:
                break
            xt = x + t * direction
            xt = np.clip(xt, -5, 5)
            ft = func(xt)
            calls[0] += 1
            if ft < best_f:
                best_f = ft
                best_t = t
        return x + best_t * direction, best_f

    def _powell_minimize(self, x0, func, calls, budget, deadline):
        """Powell's method: adaptively find minimum along rotating directions."""
        dim = self.dim
        x = np.clip(x0.copy(), -5, 5)
        fx = func(x)
        calls[0] += 1
        
        # Initial directions: coordinate axes
        directions = [np.eye(dim)[i] for i in range(dim)]
        
        max_cycles = 50
        for cycle in range(max_cycles):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            x_start_cycle = x.copy()
            fx_start_cycle = fx
            max_delta = 0.0
            best_dir_idx = 0
            
            for i in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Line search along direction[i]
                dir_i = directions[i]
                # Try a few steps
                best_t = 0.0
                best_f = fx
                for t in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.05, -0.05]:
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    xt = x + t * dir_i
                    xt = np.clip(xt, -5, 5)
                    ft = func(xt)
                    calls[0] += 1
                    if ft < best_f:
                        best_f = ft
                        best_t = t
                
                x = x + best_t * dir_i
                fx = best_f
                delta = abs(fx - fx_start_cycle)
                if delta > max_delta:
                    max_delta = delta
                    best_dir_idx = i
            
            # If no improvement, stop
            if max_delta < 1e-10:
                break
            
            # Replace the direction with the largest improvement
            # New direction = current - start of cycle
            new_dir = x - x_start_cycle
            norm = np.linalg.norm(new_dir)
            if norm > 1e-10:
                new_dir = new_dir / norm
                # Replace the direction that gave the largest improvement
                # But if it's the last direction, we can't just replace it
                # Standard Powell: replace direction[best_dir_idx] with new_dir
                # But we need to be careful about ordering
                directions[best_dir_idx] = new_dir
                # Reorder: move the new direction to the end
                if best_dir_idx != dim - 1:
                    directions.insert(dim, directions.pop(best_dir_idx))
            
            # Optional: try a line search along the new direction
            if calls[0] < budget and time.time() < deadline:
                new_dir = directions[-1]
                best_t = 0.0
                best_f = fx
                for t in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1]:
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    xt = x + t * new_dir
                    xt = np.clip(xt, -5, 5)
                    ft = func(xt)
                    calls[0] += 1
                    if ft < best_f:
                        best_f = ft
                        best_t = t
                x = x + best_t * new_dir
                fx = best_f
        
        return x, fx

    def __call__(self, func):
        deadline = time.time() + 280.0  # Safety margin
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        calls = [1]
        
        # Phase 1: (1+1)-ES for exploration (~200 calls)
        es_calls = 200
        for i in range(es_calls):
            if time.time() > deadline or calls[0] >= self.budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls[0] += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.9
            
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
        
        # Phase 2: Powell's method for refinement
        if calls[0] < self.budget and time.time() < deadline:
            x_powell, fx_powell = self._powell_minimize(
                best_x, func, calls, self.budget, deadline
            )
            if fx_powell < best_fx:
                best_x = x_powell.copy()
                best_fx = fx_powell
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
