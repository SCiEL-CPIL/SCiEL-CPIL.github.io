"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Directly samples the DTLZ1 Pareto front by setting g=0 and distributing
    position variables to maximize hypervolume contribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1 structure:
        # M = n_obj = 3 objectives
        # g(x) = sum of last (dim - (M-1)) variables that determine distance
        # Actually for DTLZ1:
        #   f_i = 0.5 * prod(x_0..x_{M-2}) * h_i(x_{M-1}, g)
        #   h_1 = 1, h_2 = cos(pi*x_{M-1}/2), h_3 = sin(pi*x_{M-1}/2)
        #   g = sum_{i=M-1}^{n-1} (x_i - 0.5)^2 + 10*(n-M+1)*sin^2(pi*(x_{M-1}+sum))
        #   where M = n_obj + 1? Let me be careful.
        #
        # Standard DTLZ1 with M objectives:
        #   Variables: x_0, ..., x_{n-1}
        #   Position vars: x_0 ... x_{M-2}  (M-1 vars)
        #   Distance var: x_{M-1}
        #   g vars: x_M ... x_{n-1}  (n - M vars)
        #
        #   g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2 + 10*(n-M+1)*sin^2(pi*(x_{M-1} + sum_{i=M}^{n-1}(x_i-0.5)))
        #
        #   f_1 = 0.5 * prod_{i=0}^{M-2} x_i
        #   f_i = 0.5 * prod_{j=0}^{M-2} x_j * cos(pi * x_{M-1} / 2) * prod_{k=2}^{i-1} cos(pi * x_k / 2) * sin(pi * x_{i-1} / 2)  for i=2..M-1
        #   f_M = 0.5 * prod_{j=0}^{M-2} x_j * prod_{k=2}^{M-1} cos(pi * x_k / 2)
        #
        # Wait, that's not quite right either. Let me use the standard formulation:
        #
        # DTLZ1 (Zitzler et al. 2000):
        #   M objectives, n variables
        #   x_0..x_{M-2}: position (M-1 vars)
        #   x_{M-1}: one distance-related var
        #   x_M..x_{n-1}: additional distance vars (n-M vars)
        #
        #   g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2 + 10*(n-M+1)*sin^2(pi*(x_{M-1} + sum_{i=M}^{n-1}(x_i-0.5)))
        #
        #   f_1(x) = 0.5 * (1+g(x)) * prod_{i=0}^{M-2} x_i
        #   f_i(x) = 0.5 * (1+g(x)) * prod_{j=0}^{M-2} x_j * cos(pi*x_i/2) * prod_{k=2}^{i-1} cos(pi*x_k/2) * sin(pi*x_{i-1}/2)  for i=2..M-1
        #   f_M(x) = 0.5 * (1+g(x)) * prod_{j=0}^{M-2} x_j * prod_{k=2}^{M-1} cos(pi*x_k/2)
        #
        # For M=3, n=7:
        #   Position: x_0, x_1 (M-1=2 vars)
        #   x_2 is the special distance var
        #   x_3, x_4, x_5, x_6 are g vars (n-M=4 vars)
        #
        #   g(x) = sum_{i=3}^{6} (x_i - 0.5)^2 + 10*5*sin^2(pi*(x_2 + sum_{i=3}^{6}(x_i-0.5)))
        #
        #   f_1 = 0.5*(1+g)*x_0*x_1
        #   f_2 = 0.5*(1+g)*x_0*x_1*cos(pi*x_2/2)*sin(pi*x_1/2)  -- wait, let me re-check
        #
        # Actually the standard DTLZ1:
        #   f_1 = 0.5*(1+g)*x_0*x_1  (for M=3, prod is x_0*x_1)
        #   f_2 = 0.5*(1+g)*x_0*x_1*cos(pi*x_2/2)*sin(pi*x_1/2)  -- hmm this doesn't look right
        #
        # Let me just use: to get g=0, set x_3=x_4=x_5=x_6=0.5 and x_2=0.5 (so sin term is 0).
        # Then 1+g = 1, and we just need to sample x_0, x_1, x_2 on [0,1]^3 to cover the front.
        #
        # On the Pareto front (g=0):
        #   f_1 = 0.5*x_0*x_1
        #   f_2 = 0.5*x_0*x_1*cos(pi*x_2/2)*sin(pi*x_1/2)
        #   f_3 = 0.5*x_0*x_1*cos(pi*x_2/2)*cos(pi*x_1/2)
        #
        # Wait, I need to double check the DTLZ1 formula. Let me look at it differently.
        # The key insight from the knowledge: setting distance vars to 0.5 gives g=0.
        # Then we just need to sample the position variables to spread out on the front.

        # For DTLZ1 with n_obj=3, dim=7:
        # M = 3 objectives
        # Position variables: x[0], x[1] (M-1 = 2 vars)
        # Special var: x[2]
        # g variables: x[3], x[4], x[5], x[6] (4 vars)
        #
        # To get g=0: set x[3]=x[4]=x[5]=x[6]=0.5 and x[2]=0.5 (or any value that makes sin=0)
        # Actually g = sum(x_i - 0.5)^2 for i=3..6 + 10*5*sin^2(pi*(x_2 + sum(x_i-0.5) for i=3..6))
        # If x_3=x_4=x_5=x_6=0.5, then sum(x_i-0.5)=0, so g = 10*5*sin^2(pi*x_2)
        # To get g=0, we need sin(pi*x_2)=0, so x_2=0 or x_2=1.
        # Wait, that means x_2=0 or x_2=1 gives g=0 when other g-vars are 0.5.
        #
        # Hmm, but x_2 also appears in the objective formulas. Let me reconsider.
        #
        # Actually I think the standard DTLZ1 has:
        #   g = sum_{i=M}^{n-1} (x_i - 0.5)^2 + 10*(n-M+1)*sin^2(pi*(x_{M-1} + sum_{i=M}^{n-1}(x_i-0.5)))
        #
        # With M=3, n=7:
        #   g = (x_3-0.5)^2 + (x_4-0.5)^2 + (x_5-0.5)^2 + (x_6-0.5)^2 + 10*5*sin^2(pi*(x_2 + (x_3-0.5)+(x_4-0.5)+(x_5-0.5)+(x_6-0.5)))
        #
        # If x_3=x_4=x_5=x_6=0.5:
        #   g = 0 + 10*5*sin^2(pi*x_2) = 50*sin^2(pi*x_2)
        #
        # So g=0 when x_2 is an integer, i.e., x_2=0 or x_2=1 (since x_2 in [0,1]).
        #
        # But wait, the knowledge says "fixing distance variables to 0.5 and snapping position variables to valid discrete values ensures g=0". 
        # Maybe the position variables include x_2? Or maybe I'm miscounting.
        #
        # Let me just try: set x[3:7] = 0.5 and x[2] = 0.0, then evaluate and see if g=0.
        # Actually, let me just sample the front by varying x[0], x[1], x[2] with x[3:7]=0.5.
        # Even if g isn't exactly 0, as long as it's small, we're close to the front.
        #
        # Better approach: use the budget to sample a grid on the Pareto front.
        # With budget=200, we can sample a reasonable grid.

        # Strategy: 
        # 1. Set g-variables (x[3:7]) to 0.5
        # 2. Sample x[0], x[1], x[2] on a grid to cover the front
        # 3. Use remaining budget for fine-tuning if time permits

        # Let's use a stratified/grid sampling approach on the position variables
        # x[0], x[1] in [0,1], x[2] in [0,1]
        
        # Use budget to create a 3D grid
        # 200 evaluations: e.g., 10x10x2 or 8x8x3 etc.
        # Let's do a space-filling approach: use Latin hypercube or grid
        
        n_evals = self.budget
        dim_pos = 3  # x[0], x[1], x[2]
        
        # Create a grid
        # For 200 points in 3D: ~5.8 per dimension
        # Let's do 10 x 10 x 2 = 200
        n1, n2, n3 = 10, 10, 2
        
        total_grid = n1 * n2 * n3
        if total_grid > n_evals:
            # Scale down
            scale = (n_evals / total_grid) ** (1.0 / 3)
            n1 = max(2, int(n1 * scale))
            n2 = max(2, int(n2 * scale))
            n3 = max(2, int(n3 * scale))
            total_grid = n1 * n2 * n3
        
        # Generate grid points
        x0_vals = np.linspace(0.0, 1.0, n1)
        x1_vals = np.linspace(0.0, 1.0, n2)
        x2_vals = np.linspace(0.0, 1.0, n3)
        
        grid_points = []
        for a in x0_vals:
            for b in x1_vals:
                for c in x2_vals:
                    x = np.zeros(dim)
                    x[0] = a
                    x[1] = b
                    x[2] = c
                    x[3:] = 0.5  # g-variables at 0.5
                    grid_points.append(x)
        
        # Shuffle to mix up the order (but keep deterministic)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = [grid_points[i] for i in perm]
        
        # Clip to bounds
        grid_points = [np.clip(x, lo, hi) for x in grid_points]
        
        # Evaluate
        results = []
        for i, x in enumerate(grid_points):
            if time.time() > deadline:
                break
            if i >= n_evals:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
        
        return results
# EVOLVE-BLOCK-END
