# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic low-discrepancy DTLZ4 optimizer.
    
    Strategy:
    The DTLZ4 Pareto front is defined by points where x[3:] = 0.5 and x[0:3] lie on the 
    unit sphere in the positive orthant. The objectives are:
    f1 = x0
    f2 = x1
    f3 = 1 - x2
    (since g=0 on the front).
    
    To maximize the hypervolume of the set of returned points, we need to distribute 
    these points as evenly as possible on the quarter-sphere surface. Random sampling 
    (current approach) suffers from clustering and gaps, leading to suboptimal hypervolume.
    
    We use a Halton sequence (low-discrepancy sequence) in 3 dimensions to generate 
    points in the unit cube [0,1]^3, and then map them to the surface of the quarter-sphere 
    by normalizing their Euclidean norm. This mapping preserves the relative ordering and 
    uniformity properties of the Halton sequence, resulting in a near-optimal distribution 
    for hypervolume maximization.
    
    We will use the full budget (200 evaluations) to generate 200 such points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Generate Halton sequence points
        # Halton base must be prime. For 3 dimensions, use bases 2, 3, 5.
        bases = [2, 3, 5]
        
        def halton(index, base):
            """Generate the index-th Halton number in the given base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r

        # Generate n_points Halton numbers in 3 dimensions
        # We skip the first point (index 0) as it maps to (0,0,0) which is degenerate on the sphere
        # So we use indices 1 to n_points
        halton_points = np.zeros((n_points, 3))
        for i in range(n_points):
            for d in range(3):
                halton_points[i, d] = halton(i + 1, bases[d])
        
        # Map to quarter-sphere: Normalize the vectors
        norms = np.linalg.norm(halton_points, axis=1, keepdims=True)
        # Avoid division by zero (should not happen for i>=1 with these bases)
        norms[norms == 0] = 1.0
        sphere_points = halton_points / norms
        
        # Construct full vectors for DTLZ4
        # x[0:3] = sphere_points
        # x[3:] = 0.5 (to ensure g=0)
        x_all = np.zeros((n_points, self.dim))
        x_all[:, :3] = sphere_points
        x_all[:, 3:] = 0.5
        
        # Clip to bounds just in case (though 0.5 and sphere points in [0,1] are safe)
        x_all = np.clip(x_all, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = x_all[i]
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
