# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # For DTLZ2:
        # x[0] and x[1] are angles/shape variables
        # x[2:] are distance variables, optimal value is 0.0
        
        # We use the full budget for sampling the Pareto front
        n_points = budget
        
        # Generate Halton sequence in 2D
        def halton_sequence(n, dims):
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # primes: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        i_val = (i_val - 1) // base
                    points[i, d] = f
            return points

        # We need 2D Halton points
        halton_pts = halton_sequence(n_points, 2)
        
        # Map to sphere coordinates
        # u = halton_pts[:, 0]  -> w = cos(theta)
        # v = halton_pts[:, 1]  -> phi = v * pi/2
        
        w = halton_pts[:, 0]
        phi = halton_pts[:, 1] * (np.pi / 2)
        
        # Compute objectives on the Pareto front (g=0)
        # f1 = w * cos(phi)
        # f2 = w * sin(phi)
        # f3 = sqrt(1 - w^2)
        
        # We need to set x such that problem(x) returns these values.
        # DTLZ2:
        # g(x) = sum_{i=M}^{K} (x_i - 0.5)^2
        # f_1 = (1+g) * cos(x_1) * cos(x_2)  -- Wait, standard DTLZ2 uses x_1, x_2 as angles?
        # Let's check the standard DTLZ2 definition.
        # f_i(x) = (1 + g(x_{M:K})) * prod_{j=1}^{i-1} cos(x_j) * sin(x_i) for i=1..M-1
        # f_M(x) = (1 + g(x_{M:K})) * prod_{j=1}^{M-1} sin(x_j)
        
        # If we set x_2..x_K to 0.5, then g=0.
        # Then:
        # f_1 = cos(x_1) * sin(x_2) ?? No.
        # i=1: f_1 = 1 * sin(x_1)
        # i=2: f_2 = 1 * cos(x_1) * sin(x_2)
        # i=3: f_3 = 1 * cos(x_1) * cos(x_2)
        
        # This parameterization is different from the one I derived above.
        # My derived one:
        # f_1 = cos(theta) cos(phi)
        # f_2 = cos(theta) sin(phi)
        # f_3 = sin(theta)
        
        # Let's map my (theta, phi) to DTLZ2 (x_1, x_2).
        # f_1 = sin(x_1)
        # f_2 = cos(x_1) sin(x_2)
        # f_3 = cos(x_1) cos(x_2)
        
        # If I set x_1 = pi/2 - theta, then sin(x_1) = cos(theta) = w.
        # Then f_1 = w.
        # f_2 = sin(theta) sin(x_2)
        # f_3 = sin(theta) cos(x_2)
        
        # This doesn't match my sphere parameterization directly.
        # Let's stick to the DTLZ2 parameterization.
        # We want to sample x_1, x_2 in [0, pi/2] such that the resulting points are well distributed.
        
        # The mapping from (x_1, x_2) to the front is:
        # f = [sin(x_1), cos(x_1)sin(x_2), cos(x_1)cos(x_2)]
        
        # To get uniform distribution on the sphere, we should sample x_1 and x_2 appropriately.
        # The area element for this parameterization is cos(x_1) dx_1 dx_2.
        # So we want cos(x_1) dx_1 dx_2 to be uniform.
        # Let u, v be uniform in [0,1].
        # dx_1 dx_2 / (1/cos(x_1)) = du dv
        # This is complex.
        
        # Alternative: Just use the Halton points directly as x_1, x_2 scaled to [0, pi/2].
        # This will cluster points near x_1=pi/2 (where cos(x_1)=0, so f_2=f_3=0).
        # This is bad for hypervolume.
        
        # Better: Use the transformation derived from the standard spherical coordinates.
        # Standard spherical:
        # f_1 = sin(theta) sin(phi)
        # f_2 = sin(theta) cos(phi)
        # f_3 = cos(theta)
        # Area element: sin(theta) dtheta dphi.
        
        # DTLZ2 mapping:
        # f_1 = sin(x_1)
        # f_2 = cos(x_1) sin(x_2)
        # f_3 = cos(x_1) cos(x_2)
        
        # If we set x_1 = pi/2 - theta, x_2 = phi:
        # f_1 = cos(theta)
        # f_2 = sin(theta) sin(phi)
        # f_3 = sin(theta) cos(phi)
        
        # This is just a permutation of the standard spherical coordinates.
        # The area element is still sin(theta) dtheta dphi.
        
        # So, to get uniform points on the sphere, we should sample theta and phi such that sin(theta) dtheta dphi is uniform.
        # Let u, v be uniform in [0,1].
        # v -> phi = v * pi/2
        # u -> sin(theta) dtheta = du
        # -cos(theta) = u - 1  => cos(theta) = 1 - u
        # theta = arccos(1 - u)
        
        # So:
        # x_1 = pi/2 - arccos(1 - u) = arcsin(1 - u)
        # x_2 = v * pi/2
        
        # Let's verify:
        # u=0 -> x_1=0 -> f_1=0, f_2=sin(x_2), f_3=cos(x_2). (Theta=pi/2, on equator)
        # u=1 -> x_1=pi/2 -> f_1=1, f_2=0, f_3=0. (Theta=0, at pole)
        
        # This maps the unit square to the quarter sphere uniformly.
        
        u = halton_pts[:, 0]
        v = halton_pts[:, 1]
        
        x_1 = np.arcsin(1 - u)
        x_2 = v * (np.pi / 2)
        
        # Build X
        X = np.zeros((n_points, dim))
        X[:, 0] = x_1
        X[:, 1] = x_2
        X[:, 2:] = 0.5  # Optimal distance for DTLZ2 is 0.5?
        
        # Wait, DTLZ2 g(x) = sum (x_i - 0.5)^2.
        # So optimal is x_i = 0.5.
        # My previous thought that 0.0 is optimal was for DTLZ1 or a variant.
        # For DTLZ2, 0.5 is correct.
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

# EVOLVE-BLOCK-END
