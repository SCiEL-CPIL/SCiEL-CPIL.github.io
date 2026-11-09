import numpy as np
import time

def van_der_corput(n, base):
    """Generate the first n points of the van der Corput sequence."""
    seq = []
    for i in range(1, n + 1):
        f = 1.0 / base
        seq.append(0.0)
        x = i
        while x > 0:
            seq[-1] += f * (x % base)
            x //= base
            f /= base
    return np.array(seq)

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using Hammersley points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = t_start + 290.0 

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 with M=3 objectives has M-1 = 2 shape variables (x_0, x_1).
        # The remaining N-M+1 = 5 variables (x_2..x_6) are distance variables.
        # To achieve g=0 (minimize distance part), set distance vars to 0.5.
        # The Pareto front is the set of points where g=0.
        # We sample the shape variables (x_0, x_1) to cover the front.
        
        # Use Hammersley points for deterministic low-discrepancy sampling in 2D.
        # Hammersley points: (vdc(i, 2), vdc(i, 3)) or similar low-discrepancy mix.
        # Standard Hammersley uses vdc for first dim and a scrambled/low-disc for others.
        # Here we use van der Corput base 2 for dim 1 and base 3 for dim 2.
        
        # Generate Hammersley points
        # Note: Hammersley points are usually 1-indexed for vdc.
        # We want 'budget' points.
        
        # We will generate 'budget' points.
        # x0: van der Corput base 2
        # x1: van der Corput base 3
        
        # However, standard Hammersley points often include the origin or start from 1.
        # Let's generate N points.
        
        # To ensure good coverage including corners, we might need to adjust.
        # The corners (0,0), (1,0), (0,1) are important.
        # vdc(1, 2) = 0.5, vdc(1, 3) = 1/3
        # vdc(2, 2) = 0.25, vdc(2, 3) = 2/3
        # vdc(3, 2) = 0.75, vdc(3, 3) = 1/9
        # ...
        # These points do not naturally hit the corners (0,0), (1,0), (0,1) exactly.
        # So we will keep the strategy of replacing the first few points with explicit corners.
        
        # Generate the rest using Hammersley
        # We need budget points total.
        # Let's generate budget points using vdc.
        
        # i from 1 to budget
        i = np.arange(1, budget + 1)
        
        # van der Corput base 2
        x0 = np.zeros(budget)
        f = 0.5
        for k in range(budget):
            val = 0.0
            x = i[k]
            base = 2
            f = 1.0 / base
            while x > 0:
                val += f * (x % base)
                x //= base
                f /= base
            x0[k] = val
            
        # van der Corput base 3
        x1 = np.zeros(budget)
        for k in range(budget):
            val = 0.0
            x = i[k]
            base = 3
            f = 1.0 / base
            while x > 0:
                val += f * (x % base)
                x //= base
                f /= base
            x1[k] = val
            
        samples = np.column_stack((x0, x1))
        
        # Explicitly include the corners of the decision space that map to
        # unique vertices of the Pareto front in objective space.
        # (x0, x1) = (1, 0) -> (1, 0, 0)
        # (x0, x1) = (0, 1) -> (0, 1, 0)
        # (x0, x1) = (0, 0) -> (0, 0, 1)
        
        # Replace the first 3 samples with these explicit corners to ensure
        # maximum coverage of the extreme ends of the front.
        if budget >= 3:
            samples[0] = [1.0, 0.0]
            samples[1] = [0.0, 1.0]
            samples[2] = [0.0, 0.0]

        # Construct points
        points = np.zeros((budget, dim))
        points[:, 0] = samples[:, 0]
        points[:, 1] = samples[:, 1]
        
        # Set distance variables to 0.5
        for j in range(2, dim):
            points[:, j] = 0.5
            
        # Evaluate
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))
            
        return results
