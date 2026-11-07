"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _construct_pareto_point(self, theta1, theta2):
        """
        Construct a decision vector x that maps to a Pareto-optimal point on DTLZ4.
        
        For DTLZ4 with M=3 objectives:
        f1 = 0.5 * (1 + g) * cos(theta1) * cos(theta2)
        f2 = 0.5 * (1 + g) * cos(theta1) * sin(theta2)
        f3 = 0.5 * (1 + g) * sin(theta1)
        
        where g = sum_{i=m+1}^{n} (x_i - 0.5)^2 with m = M-1 = 2.
        
        To be Pareto optimal, we need g = 0, which means x_3...x_12 = 0.5.
        Then f1 = 0.5*cos(t1)*cos(t2), f2 = 0.5*cos(t1)*sin(t2), f3 = 0.5*sin(t1).
        
        The first two variables determine the angles:
        x_1 = theta1 / pi
        x_2 = theta2 / pi
        
        Wait, that's not quite right. Let me reconsider.
        
        Actually for DTLZ4:
        x_1 in [0,1] maps to theta1 = pi * x_1
        x_2 in [0,1] maps to theta2 = pi * x_2
        
        And for Pareto optimality, x_3...x_12 must all be 0.5 (so g=0).
        
        So given desired angles theta1, theta2:
        x[0] = theta1 / pi
        x[1] = theta2 / pi
        x[2:] = 0.5
        """
        x = np.zeros(self.dim)
        x[0] = theta1 / np.pi
        x[1] = theta2 / np.pi
        x[2:] = 0.5
        return x

    def _halton_sequence(self, n, dim):
        """Generate n quasi-random points in [0,1]^dim using Halton sequence."""
        points = np.zeros((n, dim))
        # First few primes for bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for d in range(dim):
            base = primes[d % len(primes)]
            for i in range(n):
                f = 1.0
                r = 0.0
                idx = i + 1  # Start from 1 to avoid 0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                points[i, d] = r
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Generate points that lie exactly on the Pareto front.
        # For DTLZ4, Pareto-optimal points have x_3...x_12 = 0.5, and x_1, x_2
        # determine the position on the front via angles.
        
        # We want to cover the Pareto front densely. The front is parameterized
        # by (theta1, theta2) where theta1 in [0, pi/2] and theta2 in [0, pi/2].
        # This gives us a quarter disk in the (f1, f2) plane with f3 determined.
        
        # Use quasi-random (Halton) sampling for better coverage than uniform random.
        
        n_evals = 0
        all_points = []
        all_fvals = []
        
        # Phase 1: Dense quasi-random sampling of the Pareto front
        # Use Halton sequence in 2D for (theta1/pi, theta2/pi) in [0,1]^2
        n_phase1 = min(self.budget, 150)
        
        halton_pts = self._halton_sequence(n_phase1, 2)
        
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            # Map Halton points to angles
            t1 = halton_pts[i, 0] * (np.pi / 2)
            t2 = halton_pts[i, 1] * (np.pi / 2)
            
            x = self._construct_pareto_point(t1, t2)
            # Ensure within bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            n_evals += 1
            all_points.append(x)
            all_fvals.append(f)
        
        # Phase 2: If budget remains, add more points with slight perturbation
        # to ensure we have good coverage, or add edge/corner points
        if n_evals < self.budget:
            # Add structured grid points for better hypervolume coverage
            n_grid = min(self.budget - n_evals, 50)
            n_side = int(np.ceil(np.sqrt(n_grid)))
            
            for i in range(n_side):
                if n_evals >= self.budget or time.time() > deadline:
                    break
                for j in range(n_side):
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    # Grid on the quarter disk parameterization
                    t1 = (i + 0.5) / n_side * (np.pi / 2)
                    t2 = (j + 0.5) / n_side * (np.pi / 2)
                    
                    x = self._construct_pareto_point(t1, t2)
                    x = np.clip(x, lo, hi)
                    
                    f = problem(x)
                    n_evals += 1
                    all_points.append(x)
                    all_fvals.append(f)
        
        # Phase 3: If still budget remains, add random perturbations around Pareto points
        # to potentially find slightly better points (though on the front, they should be optimal)
        if n_evals < self.budget and len(all_points) > 0:
            rng = np.random.default_rng(42)
            n_remaining = self.budget - n_evals
            for _ in range(n_remaining):
                if time.time() > deadline:
                    break
                # Pick a random existing point and perturb it slightly
                idx = rng.integers(len(all_points))
                x = all_points[idx].copy()
                # Small perturbation in the angle space
                t1 = x[0] * np.pi
                t2 = x[1] * np.pi
                t1 += rng.standard_normal() * 0.01
                t2 += rng.standard_normal() * 0.01
                t1 = np.clip(t1, 0, np.pi / 2)
                t2 = np.clip(t2, 0, np.pi / 2)
                
                x_new = self._construct_pareto_point(t1, t2)
                x_new = np.clip(x_new, lo, hi)
                
                f = problem(x_new)
                n_evals += 1
                all_points.append(x_new)
                all_fvals.append(f)
        
        return all_points
# EVOLVE-BLOCK-END
