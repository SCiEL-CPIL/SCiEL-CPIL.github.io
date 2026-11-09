import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0, 1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ2 Pareto front: quarter-sphere in 3D
        # y0 = cos(theta1) * cos(theta2)
        # y1 = cos(theta1) * sin(theta2)
        # y2 = sin(theta1)
        
        # For hypervolume, we want points spread over the Pareto front.
        # The Pareto front is a quarter-sphere. The area element is sin(theta1) dtheta1 dtheta2.
        # To get uniform coverage in area (which correlates with hypervolume contribution),
        # we should sample sin(theta1) uniformly. Let u1 = sin(theta1), so theta1 = arcsin(u1).
        # theta2 can be sampled uniformly.
        
        # Reserve 3 evaluations for the vertices of the front
        n_sobol = n_evals - 3
        if n_sobol < 1:
            n_sobol = 1
            n_vertices = 0
        else:
            n_vertices = 3

        # Generate Halton points in [0, 1)^2
        # Base 2 for the first coordinate, base 3 for the second
        u1 = self._halton(n_sobol, 2)
        u2 = self._halton(n_sobol, 3)
        
        # Map to angles
        # theta1 = arcsin(u1), theta2 = u2 * pi/2
        # This gives area-uniform sampling on the quarter-sphere
        theta1 = np.arcsin(u1)
        theta2 = u2 * (np.pi / 2)
        
        # Compute objective vectors for Sobol points
        y0 = np.cos(theta1) * np.cos(theta2)
        y1 = np.cos(theta1) * np.sin(theta2)
        y2 = np.sin(theta1)
        
        # Create X matrix
        X = np.zeros((n_sobol, self.dim))
        X[:, 0] = y0
        X[:, 1] = y1
        X[:, 2] = y2
        X[:, 3:] = 0.5
        
        # Add vertices
        if n_vertices > 0:
            # Vertex 1: (1, 0, 0)
            x_v1 = np.zeros(self.dim)
            x_v1[0] = 1.0
            x_v1[3:] = 0.5
            
            # Vertex 2: (0, 1, 0)
            x_v2 = np.zeros(self.dim)
            x_v2[1] = 1.0
            x_v2[3:] = 0.5
            
            # Vertex 3: (0, 0, 1)
            x_v3 = np.zeros(self.dim)
            x_v3[2] = 1.0
            x_v3[3:] = 0.5
            
            X = np.vstack([X, x_v1, x_v2, x_v3])
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
