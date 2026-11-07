# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic DTLZ4 optimizer using 2D Latin Hypercube Sampling on the quarter-sphere.
    
    Strategy:
    The Pareto front of DTLZ4 with 3 objectives is the quarter-sphere defined by:
    f1^2 + f2^2 + f3^2 = 1, f_i >= 0.
    
    Instead of generating 3D points on a sphere and filtering, we parameterize the surface
    using spherical coordinates:
    f1 = sin(phi) * cos(theta)
    f2 = sin(phi) * sin(theta)
    f3 = cos(phi)
    
    Where phi is the polar angle from the z-axis (0 to pi/2) and theta is the azimuthal angle (0 to pi/2).
    
    To maximize hypervolume, we need points distributed uniformly over the surface area.
    The surface area element is proportional to sin(phi) dphi dtheta.
    Therefore, to sample uniformly on the surface, we should sample:
    u ~ Uniform(0, 1) -> phi = arccos(1 - u)  (since cos(phi) = 1 - u => phi = arccos(1-u))
    v ~ Uniform(0, 1) -> theta = v * pi/2
    
    However, the directive suggests using a Latin Hypercube Sampling (LHS) design on the 3-dimensional simplex.
    Wait, the Pareto front is a 2D manifold (sphere patch), not a 3D simplex. 
    The "simplex" usually refers to the standard simplex in R^{n_obj-1} for linear fronts, but DTLZ4 has a curved front.
    
    Let's re-read the directive: "Use a Latin Hypercube Sampling (LHS) design on the 3-dimensional simplex... reserve the final 2 evaluations to explicitly place points at the three extreme vertices".
    This phrasing is slightly confusing for DTLZ4 which has a curved front. 
    However, a common effective strategy for DTLZ4 is to sample the decision variables x[0:3] such that they lie on the unit sphere.
    
    If we interpret "3-dimensional simplex" as the domain of the first 3 variables constrained to the sphere, 
    we can use a 2D LHS on the (phi, theta) domain.
    
    Let's stick to the most robust interpretation of "uniform coverage of the Pareto front's area":
    1. Generate N-2 points using a 2D LHS on [0, 1]^2.
    2. Map (u, v) to (phi, theta) such that the resulting points are uniformly distributed on the quarter-sphere surface.
       phi = arccos(1 - u)
       theta = v * pi/2
    3. The remaining 2 evaluations (or 3, if we want 3 vertices) are placed at the extreme vertices of the front:
       (1, 0, 0), (0, 1, 0), (0, 0, 1).
       Since we have budget 200, we can place 3 vertices and 197 LHS points.
       The directive says "reserve the final 2 evaluations to explicitly place points at the three extreme vertices". 
       This is impossible with 2 evaluations for 3 vertices. 
       Perhaps it means "reserve evaluations for vertices" and the count is flexible? 
       Or maybe it implies that placing vertices is crucial. 
       Let's place the 3 vertices explicitly. This uses 3 evaluations.
       Remaining 197 evaluations for LHS.
       
    Why LHS over Halton/Fibonacci?
    LHS ensures that each marginal dimension (phi and theta) is covered exactly once in each stratum, 
    which often leads to better space-filling properties in 2D than simple random or even some low-discrepancy sequences 
    if the mapping introduces distortion.
    
    Constructing the DTLZ4 input vector:
    For a point on the Pareto front, we need x[3:] = 0.5 (to make g=0) and x[0:3] on the unit sphere.
    x[0] = sin(phi) * cos(theta)
    x[1] = sin(phi) * sin(theta)
    x[2] = cos(phi)
    x[3:] = 0.5
    
    Note: The problem clips inputs to bounds [0, 1].
    sin(phi), cos(phi) are in [0, 1].
    cos(theta), sin(theta) are in [0, 1] for theta in [0, pi/2].
    So x[0:3] will be in [0, 1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # 1. Define the 3 extreme vertices of the Pareto front (quarter sphere)
        # f1=1, f2=0, f3=0 -> x0=1, x1=0, x2=0
        # f1=0, f2=1, f3=0 -> x0=0, x1=1, x2=0
        # f1=0, f2=0, f3=1 -> x0=0, x1=0, x2=1
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # We will use 3 evaluations for vertices.
        # Remaining evaluations for LHS.
        n_lhs = n_points - 3
        
        if n_lhs < 0:
            n_lhs = 0
            vertices = vertices[:n_points] # If budget is very small, just take vertices
        
        # 2. Generate 2D Latin Hypercube Sample
        # LHS: For each dimension, divide [0,1] into n strata and pick one random point in each.
        # To be deterministic, we use a fixed permutation or just uniform random with a fixed seed?
        # The task says "np.random.seed(seed)" is called by evaluator. So we can use np.random.
        
        if n_lhs > 0:
            # Create indices for strata
            idx = np.arange(n_lhs)
            
            # Random permutation for each dimension to create the LHS
            # This ensures that in dimension 1, each stratum is hit once, and same for dimension 2.
            perm1 = np.random.permutation(n_lhs)
            perm2 = np.random.permutation(n_lhs)
            
            # Generate random offsets within strata
            # Stratum i is [i/n, (i+1)/n]
            # Point in stratum i is (i + U) / n
            u1 = (perm1 + np.random.rand(n_lhs)) / n_lhs
            u2 = (perm2 + np.random.rand(n_lhs)) / n_lhs
            
            # Map to phi and theta
            # u1 -> phi. We want uniform area on sphere.
            # Area element ~ sin(phi) dphi.
            # Let C = cos(phi). dC = -sin(phi) dphi.
            # So uniform C in [0, 1] gives uniform area.
            # C = 1 - u1 (so u1=0 -> C=1 -> phi=0, u1=1 -> C=0 -> phi=pi/2)
            c_phi = 1.0 - u1
            # s_phi = sqrt(1 - c_phi^2)
            s_phi = np.sqrt(1.0 - c_phi**2)
            
            # u2 -> theta. Uniform in [0, pi/2]
            theta = u2 * (np.pi / 2.0)
            
            # Calculate coordinates
            x0 = s_phi * np.cos(theta)
            x1 = s_phi * np.sin(theta)
            x2 = c_phi
            
            lhs_points = np.column_stack((x0, x1, x2))
        else:
            lhs_points = np.empty((0, 3))
        
        # 3. Combine vertices and LHS points
        all_points = np.vstack((vertices[:min(3, n_points)], lhs_points))
        
        # 4. Construct full decision vectors
        x_all = np.zeros((len(all_points), self.dim))
        x_all[:, :3] = all_points
        x_all[:, 3:] = 0.5
        
        # Clip to bounds
        x_all = np.clip(x_all, lo, hi)
        
        # 5. Evaluate
        results = []
        for i in range(len(x_all)):
            if time.time() > deadline:
                break
            x = x_all[i]
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
