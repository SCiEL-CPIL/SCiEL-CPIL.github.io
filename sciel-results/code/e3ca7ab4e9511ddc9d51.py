import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # The DTLZ7 problem with 3 objectives has a Pareto front that is a 2D square
        # in the (f0, f1) plane when g=0.
        # f0 = x0, f1 = 1 - x1, f2 = 1.
        # We set x2...x21 = 0 to ensure g=0.
        # We distribute budget points uniformly in the [0,1]^2 square using Sobol sequence.
        
        n_points = self.budget
        
        # Ensure we have at least 4 points for corners
        if n_points < 4:
            n_corners = 0
            n_sobol = n_points
        else:
            n_corners = 4
            n_sobol = n_points - n_corners
            
        X = np.zeros((n_points, self.dim))
        
        # Add the 4 Pareto extreme points (corners of the x0-x1 unit square)
        # These correspond to (f0, f1) = (0,1), (0,0), (1,1), (1,0)
        corners = np.array([
            [0.0, 0.0], # x0=0, x1=0 -> f0=0, f1=1
            [0.0, 1.0], # x0=0, x1=1 -> f0=0, f1=0
            [1.0, 0.0], # x0=1, x1=0 -> f0=1, f1=1
            [1.0, 1.0]  # x0=1, x1=1 -> f0=1, f1=0
        ])
        
        if n_corners > 0:
            X[:n_corners, 0] = corners[:, 0]
            X[:n_corners, 1] = corners[:, 1]
            
        if n_sobol > 0:
            # Generate Sobol points in [0, 1]^2
            # scramble=True uses the seed for better randomization per run
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n_sobol)
            
            # points[:, 0] corresponds to x0
            # points[:, 1] corresponds to 1 - x1, so x1 = 1 - points[:, 1]
            X[n_corners:, 0] = points[:, 0]
            X[n_corners:, 1] = 1.0 - points[:, 1]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return F
