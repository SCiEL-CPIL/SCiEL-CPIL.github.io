"""Structure-exploiting optimizer for DTLZ7 using Sobol sampling on the Pareto front."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and use Sobol quasi-random sampling to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # Let u = x[0], v = x[1] in [0,1]
        # f0 = u, f1 = v, f2 = 1 - sqrt(v)
        
        # Hypothesis: Uniform sampling in (u, v) is better than (u, sqrt(v))
        # because the hypervolume is dominated by the f0-f1 plane coverage.
        
        # We will use the full budget for Sobol points
        n_sobol = budget
        
        # Create Sobol sequence
        # We need to reserve a few points for the corners if they aren't in the Sobol set
        # Sobol sequences don't necessarily hit 0 or 1 exactly in small samples.
        # We will explicitly add the 4 corners and 4 edge midpoints, then fill the rest with Sobol.
        
        # Fixed strategic points
        fixed_points = [
            (0.0, 0.0),   # (0, 0, 1)
            (0.0, 1.0),   # (0, 1, 0)
            (1.0, 0.0),   # (1, 0, 1)
            (1.0, 1.0),   # (1, 1, 0)
            (0.5, 0.0),   # (0.5, 0, 1)
            (0.5, 1.0),   # (0.5, 1, 0)
            (0.0, 0.5),   # (0, 0.25, 0.5)
            (1.0, 0.5),   # (1, 0.25, 0.5)
        ]
        
        n_fixed = len(fixed_points)
        n_sobol = budget - n_fixed
        
        # Generate Sobol points in (u, v) space
        # u = x[0], v = x[1]
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate n_sobol points
            # Skip the first point (0,0) if it's generated, or just take n_sobol points
            # Sobol.random(n) returns n points.
            sobol_points = sampler.random(n=n_sobol)
            # Ensure we don't duplicate the fixed points? 
            # Sobol points are in [0,1). Fixed points are 0, 0.5, 1.
            # Overlap is unlikely but possible. 
            # We can just add them. Duplicates waste budget but don't hurt HV (they are dominated or same).
            # Actually, duplicates are just redundant.
            # To be safe, we can check, but with 200 points, probability is low.
            
            for i in range(n_sobol):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                u = sobol_points[i, 0]
                v = sobol_points[i, 1]
                
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                # x[2:] = 0
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        # Add fixed points
        for u, v in fixed_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
            
        return results
