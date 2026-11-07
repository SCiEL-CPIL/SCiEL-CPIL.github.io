"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic scan + local coordinate descent for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, dim, num_points):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = []
        primes = self._first_primes(dim)
        for i in range(1, num_points + 1):
            point = []
            for j in range(dim):
                p = primes[j]
                f = 1.0
                ip = i
                while ip > 0:
                    f /= p
                    ip = int(ip / p)
                point.append(f)
            points.append(point)
        return np.array(points)

    def _first_primes(self, count):
        """Return the first 'count' prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < count:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Phase 1: Global Scan using Halton sequence
        # We allocate 60% of the budget to the global scan to ensure good coverage
        scan_budget = int(budget * 0.6)
        
        # Generate scan points
        # Halton sequence is deterministic, so we don't need a random seed
        # Map [0, 1]^dim to [-5, 5]^dim
        halton_points = self._halton_sequence(dim, scan_budget)
        scan_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Evaluate scan points
        for i in range(scan_budget):
            if time.time() > deadline or calls >= budget:
                break
            
            x = scan_points[i]
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local Coordinate Descent Refinement
        # Use the remaining budget to refine the best point found
        remaining_budget = budget - calls
        current_x = best_x.copy()
        current_f = best_f
        
        # Initial step size for local search
        step_size = 1.0
        min_step_size = 1e-4
        
        # Coordinate descent loop
        while calls < budget and time.time() < deadline:
            improved = False
            # Try each dimension
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_pos = current_x.copy()
                x_pos[d] += step_size
                x_pos = np.clip(x_pos, -5, 5)
                f_pos = func(x_pos)
                calls += 1
                
                # Try negative step
                x_neg = current_x.copy()
                x_neg[d] -= step_size
                x_neg = np.clip(x_neg, -5, 5)
                f_neg = func(x_neg)
                calls += 1
                
                # Select the best move
                if f_pos < current_f:
                    current_x = x_pos
                    current_f = f_pos
                    improved = True
                    if f_pos < best_f:
                        best_f = f_pos
                        best_x = current_x.copy()
                elif f_neg < current_f:
                    current_x = x_neg
                    current_f = f_neg
                    improved = True
                    if f_neg < best_f:
                        best_f = f_neg
                        best_x = current_x.copy()
            
            # If no improvement in any dimension, shrink step size
            if not improved:
                step_size *= 0.5
                if step_size < min_step_size:
                    break
            else:
                # If improved, we can keep the step size or slightly increase it?
                # Usually keep it to ensure stability, or increase if we are far from optimum.
                # For step functions, keeping it is safer.
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
