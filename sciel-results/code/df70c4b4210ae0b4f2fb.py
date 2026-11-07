"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for WFG9 using a constrained random search strategy.
    Samples the decision space exclusively within a hyper-sphere centered at the origin 
    (where the WFG9 Pareto front is located) with a radius that shrinks linearly over the budget.
    This bypasses deceptive intermediate regions without relying on evolutionary operators.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Compute the center and half-widths for mapping from unit sphere to box
        center = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0

        # We will sample points in a hyper-sphere centered at the origin of the decision space.
        # However, the "origin" in the context of WFG9's Pareto front structure is often 
        # related to the shape variables being 0. 
        # The directive says "centered at the origin". In WFG, variables are typically in [0,1].
        # If we center at the origin (0,0,...,0), we are sampling in the corner.
        # But the Pareto front of WFG9 is not at the corner (0,0,...,0) in the decision space.
        # The Pareto front is defined by specific relationships.
        
        # Let's re-read the directive carefully: 
        # "samples the decision space exclusively within a hyper-sphere centered at the origin 
        # (where the WFG9 Pareto front is located) with a radius that shrinks linearly over the budget"
        
        # This implies the Pareto front is *near* the origin. 
        # In WFG9, if we set shape variables to 0, the position variables define the front.
        # If the bounds are [0,1], the origin is (0,0,...,0).
        # If the front is near the origin, sampling near the origin makes sense.
        
        # Let's generate points in a hyper-sphere of radius R(t) centered at the origin (0,0,...,0).
        # R(t) starts large (e.g., sqrt(dim) to cover the box) and shrinks to 0.
        # Actually, if the front is at the origin, we should start small?
        # "radius that shrinks linearly over the budget"
        
        # Let's assume the "origin" refers to the point (0, 0, ..., 0) in the decision space.
        # We generate random directions and scale them by a radius.
        
        n_eval = 0
        results = []
        
        # Pre-generate a batch of random directions to speed up numpy operations
        # We will process in chunks to respect the time limit and memory.
        chunk_size = 50
        n_chunks = (self.budget + chunk_size - 1) // chunk_size
        
        # Initial radius: sqrt(dim) ensures the sphere covers the [0,1]^dim box if centered at 0?
        # No, distance from (0,0) to (1,1) is sqrt(2). For dim=12, sqrt(12) ~ 3.46.
        # The box [0,1]^12 is contained in a sphere of radius sqrt(12) centered at (0,0,...,0)?
        # Yes, the furthest point is (1,1,...,1) with distance sqrt(12).
        
        # Final radius: 0 (converge to origin)
        
        for chunk_idx in range(n_chunks):
            if time.time() > deadline:
                break
            if n_eval >= self.budget:
                break
                
            current_chunk = min(chunk_size, self.budget - n_eval)
            
            # Linearly shrink radius from R_max to 0
            # t is fraction of budget used
            t = n_eval / self.budget
            # R_max = sqrt(dim)
            R_max = np.sqrt(self.dim)
            R_current = R_max * (1.0 - t)
            
            # Generate random points in hyper-sphere
            # 1. Generate random normal vectors
            Z = np.random.randn(current_chunk, self.dim)
            # 2. Normalize to get direction on unit sphere
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            directions = Z / norms
            
            # 3. Scale by radius
            # To sample uniformly within the sphere, we need to scale by R * u^(1/dim)
            # where u is uniform random in [0,1]
            u = np.random.rand(current_chunk, 1)
            radii = R_current * (u ** (1.0 / self.dim))
            
            X_chunk = directions * radii
            
            # Clip to bounds? The problem statement says problem(x) clips to bounds.
            # So we can pass any finite vector.
            # However, sampling outside [0,1] might be wasteful if the front is inside.
            # But the directive says "within a hyper-sphere centered at the origin".
            # If the origin is (0,0,...,0), the sphere extends into negative values.
            # WFG9 variables are usually positive.
            
            for i in range(current_chunk):
                if time.time() > deadline:
                    break
                if n_eval >= self.budget:
                    break
                try:
                    f = problem(X_chunk[i])
                    results.append(f)
                    n_eval += 1
                except Exception:
                    # Stop if an exception occurs
                    return results
                    
        return results
# EVOLVE-BLOCK-END
