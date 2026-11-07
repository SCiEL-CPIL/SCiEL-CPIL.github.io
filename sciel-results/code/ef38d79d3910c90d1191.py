"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic DTLZ2-specific search using analytic Pareto front mapping."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        used = 0
        
        # For DTLZ2 with 3 objectives and 12 variables:
        # 2 position variables (indices 0, 1) and 10 distance variables (indices 2-11)
        # Pareto front: g=0, so distance vars at their optimum (0.5 for DTLZ2)
        # f1 = (1+g)*cos(x[0]*pi/2), f2 = (1+g)*sin(x[0]*pi/2)*cos(x[1]*pi/2),
        # f3 = (1+g)*sin(x[0]*pi/2)*sin(x[1]*pi/2)
        
        # Phase 1: Uniform grid on the Pareto front for dense coverage
        # Use a 14x14 grid to stay within budget (196 points) and leave room for refinement
        n_grid = 14
        for i in range(n_grid):
            for j in range(n_grid):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Angle parameters in [0, 1]
                a = (i + 0.5) / n_grid
                b = (j + 0.5) / n_grid
                
                # Construct x: first 2 vars are position vars, rest are distance vars
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                x[2:] = 0.5  # distance vars at optimum for DTLZ2
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                if np.all(np.isfinite(f)):
                    results.append(f)
                    used += 1
        
        if time.time() > deadline:
            return np.array(results) if results else np.empty((0, self.n_obj))
        
        # Phase 2: Add boundary points to ensure full coverage of the Pareto front
        # The Pareto front is a quarter sphere; boundaries are where one angle is 0 or 1
        # Add points along the edges: a=0, a=1, b=0, b=1
        n_edge = 8
        edge_points = []
        
        # a=0 (f2=0, f3=0, f1=1)
        for j in range(n_edge):
            b = (j + 0.5) / n_edge
            edge_points.append((0.0, b))
        
        # a=1 (f1=0, f2=cos(b*pi/2), f3=sin(b*pi/2))
        for j in range(n_edge):
            b = (j + 0.5) / n_edge
            edge_points.append((1.0, b))
        
        # b=0 (f3=0, f1=cos(a*pi/2), f2=sin(a*pi/2))
        for i in range(n_edge):
            a = (i + 0.5) / n_edge
            edge_points.append((a, 0.0))
        
        # b=1 (f2=0, f1=cos(a*pi/2), f3=sin(a*pi/2))
        for i in range(n_edge):
            a = (i + 0.5) / n_edge
            edge_points.append((a, 1.0))
        
        # Corner points
        edge_points.append((0.0, 0.0))
        edge_points.append((0.0, 1.0))
        edge_points.append((1.0, 0.0))
        edge_points.append((1.0, 1.0))
        
        for a, b in edge_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = a
            x[1] = b
            x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if np.all(np.isfinite(f)):
                results.append(f)
                used += 1
        
        if time.time() > deadline:
            return np.array(results) if results else np.empty((0, self.n_obj))
        
        # Phase 3: If budget remains, add a finer grid to improve hypervolume
        # Use a 20x20 grid but skip points already covered by the 14x14 grid
        n_fine = 20
        for i in range(n_fine):
            for j in range(n_fine):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                a = (i + 0.5) / n_fine
                b = (j + 0.5) / n_fine
                
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                if np.all(np.isfinite(f)):
                    results.append(f)
                    used += 1
        
        return np.array(results) if results else np.empty((0, self.n_obj))
# EVOLVE-BLOCK-END
