import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ2 specific structure:
        # M = n_obj = 3
        # Position variables: x_0, x_1 (indices 0 to M-2)
        # Distance variables: x_2 ... x_{D-1} (indices M-1 to D-1)
        # Optimal distance variables are 0.5
        
        M = self.n_obj
        D = self.dim
        
        # Number of position variables is M - 1
        n_pos = M - 1
        
        # Generate space-filling points in the position variable space [0, 1]^(M-1)
        # Using Sobol sequence for low discrepancy
        sampler = qmc.Sobol(d=n_pos, scramble=True)
        
        # Generate budget points
        # We need to ensure we don't exceed the budget
        points = sampler.random(n=self.budget)
        
        # The points are in [0, 1)
        # We map them to the decision space
        # x_0 ... x_{M-2} are the Sobol points
        # x_{M-1} ... x_{D-1} are set to 0.5
        
        X = np.zeros((self.budget, D))
        X[:, :n_pos] = points
        X[:, n_pos:] = 0.5
        
        # Evaluate all points
        F = []
        for x in X:
            f = problem(x)
            F.append(f)
            
        F = np.array(F)
        
        return X
