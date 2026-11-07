<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Rank-1 Covariance Adaptation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        sigma = 2.0
        C = np.eye(self.dim)

        calls = 1

        while calls < self.budget:
            if time.time() > deadline:
                break

            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)

            z = rng.standard_normal(self.dim)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()

                # Update covariance towards the successful direction in standard space
                mu_c = 0.15
                C = (1 - mu_c) * C + mu_c * np.outer(z, z)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)

                # Adapt sigma based on step norm
                step_norm = np.linalg.norm(step)
                if step_norm > sigma * 1.2:
                    sigma *= 1.08
                elif step_norm < sigma * 0.5:
                    sigma *= 0.95
                else:
                    sigma *= 1.02
                sigma = np.clip(sigma, 1e-4, 10.0)
            else:
                # Failed step: shrink sigma
                sigma *= 0.92
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Occasionally re-center C to avoid drift
                if calls % 25 == 0:
                    C = 0.85 * C + 0.15 * np.eye(self.dim)
                    trace_C = np.trace(C)
                    if trace_C > 1e-10:
                        C = C * (self.dim / trace_C)

            # Periodic eigenvalue clipping for stability
            if calls % 100 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.05, 20.0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)

        return best_x, best_fx
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-wise sequential search exploiting Bent Cigar structure."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        calls = 1
        dim = self.dim

        # Golden section ratio
        gr = (np.sqrt(5.0) - 1.0) / 2.0

        # Phase 1: Coordinate-wise line search with golden section
        # For each coordinate, do a 1D golden section search
        # This exploits the Bent Cigar structure where one dimension is highly conditioned

        # First, find the best coordinate to optimize
        # Start with a broad exploration
        for d in range(dim):
            if calls >= self.budget or time.time() > deadline:
                break
            # Try a few steps in each direction
            for delta in [1.0, 0.5, 0.25, -1.0, -0.5, -0.25]:
                if calls >= self.budget or time.time() > deadline:
                    break
                y = x.copy()
                y[d] = np.clip(y[d] + delta, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_fx:
                        best_fx = fy
                        best_x = x.copy()

        # Phase 2: Iterative coordinate-wise golden section refinement
        # Cycle through dimensions, doing 1D optimization on each
        step_size = 2.0
        for iteration in range(50):
            if calls >= self.budget or time.time() > deadline:
                break
            
            for d in range(dim):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Golden section search on dimension d
                # Initial bracket: x[d] ± step_size
                a = np.clip(x[d] - step_size, -5, 5)
                b = np.clip(x[d] + step_size, -5, 5)
                
                if a >= b:
                    continue
                
                # Golden section points
                c = b - gr * (b - a)
                d_pt = a + gr * (b - a)
                
                # Evaluate at c and d_pt
                x_c = x.copy()
                x_c[d] = c
                x_d = x.copy()
                x_d[d] = d_pt
                
                if calls >= self.budget or time.time() > deadline:
                    break
                fc = func(x_c)
                calls += 1
                
                if calls >= self.budget or time.time() > deadline:
                    break
                fd = func(x_d)
                calls += 1
                
                # Iterate golden section for up to 5 iterations
                for _ in range(5):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    if fc < fd:
                        b = d_pt
                        d_pt = c
                        fd = fc
                        c = b - gr * (b - a)
                        x_c = x.copy()
                        x_c[d] = c
                        if calls >= self.budget or time.time() > deadline:
                            break
                        fc = func(x_c)
                        calls += 1
                    else:
                        a = c_pt
                        c = d_pt
                        fc = fd
                        d_pt = a + gr * (b - a)
                        x_d = x.copy()
                        x_d[d] = d_pt
                        if calls >= self.budget or time.time() > deadline:
                            break
                        fd = func(x_d)
                        calls += 1
                
                # Take the best point found
                if fc < fd:
                    new_val = c
                    new_fx = fc
                else:
                    new_val = d_pt
                    new_fx = fd
                
                if new_fx < fx:
                    x[d] = new_val
                    fx = new_fx
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                
                # Adapt step size based on progress
                if new_fx < fx:
                    step_size *= 0.7
                else:
                    step_size *= 1.3
                
                step_size = np.clip(step_size, 1e-4, 5.0)
            
            # If step size is very small, we're converging
            if step_size < 1e-3:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I need to fix a bug in my golden section search - I used `c_pt` which is undefined. Let me correct this:

