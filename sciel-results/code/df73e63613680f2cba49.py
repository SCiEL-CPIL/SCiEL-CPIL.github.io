"""Lattice-based deterministic grid search exploiting Rastrigin's periodic structure."""
import numpy as np
import time


class Optimizer:
    """Lattice-based optimizer for 5D Rastrigin with structured grid search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = [0]
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal best_x, best_f
            calls[0] += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has periodic structure with minima at integer coordinates (before shift/rotation).
        # We systematically evaluate points on a lattice to rapidly identify the global basin.
        
        # Generate lattice points: integer coordinates and half-integer offsets
        # For 5D, we need to be selective about which lattice points to evaluate
        lattice_points = []
        
        # Core lattice: all integer points in [-3, 3]^5 (too many for full grid)
        # Instead, use a stratified approach:
        # 1) Origin and axis-aligned integer points
        # 2) Half-integer offsets from best regions
        # 3) Sparse 3D slices of the 5D integer grid
        
        # Origin
        lattice_points.append(np.zeros(dim))
        
        # Integer points along each axis (sparse)
        for d in range(dim):
            for v in range(-4, 5):
                pt = np.zeros(dim)
                pt[d] = float(v)
                lattice_points.append(pt)
        
        # Half-integer offsets from origin
        for d in range(dim):
            for offset in [0.5, -0.5]:
                pt = np.zeros(dim)
                pt[d] = offset
                lattice_points.append(pt)
        
        # 2D slices: pairs of dimensions with integer values
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for v1 in range(-2, 3):
                    for v2 in range(-2, 3):
                        pt = np.zeros(dim)
                        pt[d1] = float(v1)
                        pt[d2] = float(v2)
                        lattice_points.append(pt)
        
        # 3D slices: triples of dimensions with integer values (sparse)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for d3 in range(d2 + 1, dim):
                    for v1 in range(-1, 2):
                        for v2 in range(-1, 2):
                            for v3 in range(-1, 2):
                                pt = np.zeros(dim)
                                pt[d1] = float(v1)
                                pt[d2] = float(v2)
                                pt[d3] = float(v3)
                                lattice_points.append(pt)
        
        # Add some random points for diversity
        for _ in range(20):
            lattice_points.append(rng.uniform(-4, 4, dim))
        
        # Shuffle for diversity
        for i in range(len(lattice_points) - 1, 0, -1):
            j = rng.integers(0, i + 1)
            lattice_points[i], lattice_points[j] = lattice_points[j], lattice_points[i]
        
        # Evaluate lattice points (up to 150 calls)
        lattice_budget = 150
        for x in lattice_points:
            if calls[0] >= lattice_budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            call_func(x_clipped)
        
        # Phase 2: Local refinement around best lattice point using (1+1)-ES
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            no_improve = 0
            max_no_improve = 30
            
            while calls[0] < budget and time.time() < deadline:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                if sigma < 0.003:
                    break
                if no_improve >= max_no_improve:
                    break
        
        # Phase 3: Multi-start refinement from diverse lattice points
        # Pick top-3 distinct lattice points and refine each
        # (We'll just do a few more random restarts from near the best)
        remaining = budget - calls[0]
        if remaining > 50 and best_x is not None:
            for restart in range(min(3, remaining // 50)):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Start near best but with perturbation
                start_x = best_x + 0.5 * rng.standard_normal(dim)
                start_x = np.clip(start_x, -5, 5)
                fx = call_func(start_x)
                x = start_x.copy()
                
                sigma = 0.2
                no_improve = 0
                max_no_improve = 20
                
                while calls[0] < budget and time.time() < deadline:
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = call_func(y)
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.1
                        no_improve = 0
                    else:
                        sigma *= 0.9
                        no_improve += 1
                    
                    if sigma < 0.002 or no_improve >= max_no_improve:
                        break
        
        return best_x, best_f
