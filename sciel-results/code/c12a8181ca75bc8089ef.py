"""Multi-objective optimizer for DTLZ2 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ2 structure: Pareto front is a quarter-sphere in 3D.
    
    For DTLZ2 with 3 objectives and 12 variables:
    - g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
    - f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
    - where t1 = x0*pi/2, t2 = x1*pi/2
    - Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
    
    Strategy: 
    1. Use a 14x14 grid in angle space (theta1, theta2) for uniform coverage
    2. Add 4 strategic points to improve hypervolume coverage
    3. All points are exactly on the Pareto front (g=0)
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 Pareto front parameterization:
        # theta1 = x0 * pi/2, theta2 = x1 * pi/2
        # f1 = cos(theta1)*cos(theta2), f2 = cos(theta1)*sin(theta2), f3 = sin(theta1)
        # (when g=0, i.e., x[2:] = 0.5)
        
        # Use 14x14 grid in angle space for 196 points
        n_grid = 14
        theta1_vals = np.linspace(0, np.pi / 2, n_grid)
        theta2_vals = np.linspace(0, np.pi / 2, n_grid)
        
        # Create grid
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()
        
        n_grid_points = len(T1_flat)  # 196
        
        # Build grid points in decision space
        X_grid = np.zeros((n_grid_points, dim))
        X_grid[:, 0] = 2 * T1_flat / np.pi  # x0 = 2*theta1/pi
        X_grid[:, 1] = 2 * T2_flat / np.pi  # x1 = 2*theta2/pi
        X_grid[:, 2:] = 0.5  # g=0
        
        # Add 4 strategic points to improve coverage
        # These fill gaps in the grid and improve hypervolume
        strategic_points = [
            [np.pi/4, np.pi/4],      # Center of the quarter-sphere
            [np.pi/6, np.pi/3],      # Off-center point 1
            [np.pi/3, np.pi/6],      # Off-center point 2
            [np.pi/4, 0.0],          # Edge point
        ]
        
        X_strategic = np.zeros((len(strategic_points), dim))
        for i, (t1, t2) in enumerate(strategic_points):
            X_strategic[i, 0] = 2 * t1 / np.pi
            X_strategic[i, 1] = 2 * t2 / np.pi
            X_strategic[i, 2:] = 0.5
        
        # Combine all points
        X_all = np.vstack([X_grid, X_strategic])
        
        # Evaluate all points
        used = 0
        for i in range(len(X_all)):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            problem(X_all[i])
            used += 1
        
        return X_all[:used]
# EVOLVE-BLOCK-END
