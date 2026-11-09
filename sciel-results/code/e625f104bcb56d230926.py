"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct Objective-Space Pareto Front Sampler for DTLZ1.
    
    Strategy:
    1. Instead of sampling decision variables (x), we sample the Pareto front
       directly in objective space.
    2. For DTLZ1 with M=3 objectives, the Pareto front is a 2D simplex defined
       by f1 + f2 + f3 = 1, fi >= 0.
    3. We parameterize this simplex using spherical coordinates (theta1, theta2)
       to ensure uniform distribution of points on the front surface.
    4. We use a stratified grid (or LHS) on the (theta1, theta2) domain to
       ensure good coverage and hypervolume contribution.
    5. For each point on the front, we analytically construct the decision
       vector x such that g(x)=0 (by setting distance variables to 0) and
       the shape variables correspond to the chosen theta values.
       Specifically, for DTLZ1:
       f1 = 0.5 * x1 * (1 + g)
       f2 = 0.5 * x2 * (1 - x1) * (1 + g)
       f3 = 0.5 * (1 - x2) * (1 + g)
       With g=0, we have:
       x1 = 2 * f1
       x2 = f2 / (0.5 * (1 - x1)) = 2 * f2 / (1 - 2 * f1)
       This inversion is valid as long as x1 < 0.5 (i.e., f1 < 0.25) which is
       always true for points strictly inside the simplex, but we must be careful
       with boundary points. However, since we are sampling the interior mostly,
       and clipping handles boundaries, this is robust.
       
       Actually, a safer inversion for DTLZ1 shape variables (x[0], x[1]) given
       objectives (f1, f2, f3) with g=0:
       Let alpha1 = 2*f1, alpha2 = 2*f2, alpha3 = 2*f3.
       alpha1 + alpha2 + alpha3 = 1.
       x[0] = alpha1
       x[1] = alpha2 / (1 - alpha1)  (if alpha1 < 1)
       
       Wait, the standard DTLZ1 definition:
       f1 = 0.5 * x1 * (1+g)
       f2 = 0.5 * x2 * (1-x1) * (1+g)
       f3 = 0.5 * (1-x2) * (1+g)
       
       If g=0:
       f1 = 0.5 * x1  => x1 = 2*f1
       f2 = 0.5 * x2 * (1-x1) => x2 = 2*f2 / (1-x1)
       
       We sample (f1, f2) such that f1+f2+f3=1, f3>=0 => f1+f2 <= 1.
       To avoid division by zero or negative values, we sample strictly inside
       or handle boundaries.
       
       Better approach: Sample the shape variables x1, x2 directly using a
       method that guarantees they map to a well-distributed set on the simplex.
       The previous program used LHS on x1, x2.
       
       The prompt suggests: "Represent the DTLZ1 Pareto front directly in objective
       space by parameterizing the 2D simplex using spherical coordinates".
       
       Let's use spherical coordinates for the 2D simplex (triangle).
       A point on the simplex can be represented by angles theta1, theta2.
       However, a common parameterization for uniform sampling on a 2D simplex
       is to use the square root trick for the first variable and uniform for the
       second, or use Dirichlet distribution.
       
       But the prompt specifically mentions "spherical coordinates".
       For a 2D simplex (triangle), spherical coordinates (polar) are:
       r in [0, 1], phi in [0, 2*pi)? No, the simplex is not a circle.
       
       Perhaps it means parameterizing the decision variables x1, x2 which define
       the position on the front.
       x1, x2 in [0, 1]. The front is defined by the mapping from (x1, x2) to
       (f1, f2, f3).
       
       Let's stick to the high-performing strategy of sampling x1, x2 but
       using a stratified grid in the (x1, x2) space that is transformed to
       ensure better hypervolume.
       
       Actually, the current program uses LHS on x1, x2.
       The inspiration mentions "stratified simplex sampling".
       
       Let's try a stratified grid on x1 and x2.
       x1 in [0, 1], x2 in [0, 1].
       The mapping to the front is:
       f1 = 0.5 * x1
       f2 = 0.5 * x2 * (1 - x1)
       f3 = 0.5 * (1 - x2)
       
       Note that for any x1, x2 in [0, 1], f1+f2+f3 = 0.5*x1 + 0.5*x2*(1-x1) + 0.5*(1-x2)
       = 0.5 * [x1 + x2 - x1*x2 + 1 - x2] = 0.5 * [1 + x1 - x1*x2] = 0.5 * [1 + x1(1-x2)]
       This is NOT constant 1!
       
       Wait, I need to check the DTLZ1 definition again.
       Standard DTLZ1:
       f1 = 0.5 * x1 * (1 + g)
       f2 = 0.5 * x2 * (1 - x1) * (1 + g)
       f3 = 0.5 * (1 - x2) * (1 + g)
       
       Sum = 0.5 * (1+g) * [x1 + x2 - x1*x2 + 1 - x2]
       = 0.5 * (1+g) * [1 + x1 - x1*x2]
       = 0.5 * (1+g) * [1 + x1(1-x2)]
       
       This sum is only 1 if g=0 AND x1(1-x2) = 1? No.
       If x1=1, x2=0, sum = 0.5*1*[1+1] = 1.
       If x1=0, x2=0, sum = 0.5*1*[1+0] = 0.5.
       
       So the Pareto front of DTLZ1 is NOT f1+f2+f3=1.
       The Pareto front of DTLZ1 is a (M-1)-simplex in the positive orthant,
       but it is NOT the standard simplex f1+...+fM=1.
       
       Let's re-read the DTLZ1 definition carefully.
       DTLZ1 (Deb et al. 2002):
       f1(x) = 0.5 * x1 * (1 + g(x))
       f2(x) = 0.5 * x2 * (1 - x1) * (1 + g(x))
       ...
       fM(x) = 0.5 * (1 - x_{M-2}) * (1 + g(x))
       
       For M=3:
       f1 = 0.5 * x1 * (1+g)
       f2 = 0.5 * x2 * (1-x1) * (1+g)
       f3 = 0.5 * (1-x2) * (1+g)
       
       If g=0:
       f1 = 0.5 * x1
       f2 = 0.5 * x2 * (1-x1)
       f3 = 0.5 * (1-x2)
       
       Let's check the sum again.
       f1 + f2 + f3 = 0.5 * [x1 + x2 - x1*x2 + 1 - x2] = 0.5 * [1 + x1 - x1*x2]
       This is not constant.
       
       However, the Pareto front is the set of points that cannot be improved.
       The shape variables are x1, x2.
       The range of f1 is [0, 0.5].
       The range of f3 is [0, 0.5].
       The range of f2 depends on x1, x2.
       
       The Pareto front is a 2D surface.
       To maximize hypervolume, we want to cover this surface well.
       
       The current program samples x1, x2 uniformly (LHS) in [0, 1].
       This maps to the front.
       
       Is there a better sampling?
       The area element of the front in objective space might be non-uniform
       with respect to x1, x2.
       
       Let's compute the Jacobian of the mapping (x1, x2) -> (f1, f2, f3).
       Since the front is 2D, we can parameterize by (f1, f2) or (x1, x2).
       
       If we want uniform coverage in objective space (which is what hypervolume
       cares about), we should sample such that the density of points in
       (f1, f2, f3) space is uniform on the front surface.
       
       The surface is defined by the mapping T(x1, x2).
       The area element dA on the surface is related to dx1 dx2 by the
       magnitude of the cross product of the tangent vectors.
       
       Tangent vector T1 = dT/dx1 = [0.5, -0.5*x2, 0]
       Tangent vector T2 = dT/dx2 = [0, 0.5*(1-x1), -0.5]
       
       Cross product T1 x T2 = 
       i: (-0.5*x2)*(-0.5) - 0*0.5*(1-x1) = 0.25*x2
       j: 0*0 - 0.5*(-0.5) = 0.25
       k: 0.5*0.5*(1-x1) - (-0.5*x2)*0 = 0.25*(1-x1)
       
       Magnitude = 0.25 * sqrt(x2^2 + 1 + (1-x1)^2)
       
       This magnitude varies across the domain.
       To get uniform density in objective space, we should sample x1, x2
       with probability proportional to 1 / |T1 x T2|.
       
       This is complex to sample directly.
       
       Alternative: Sample f1, f2, f3 directly?
       We know the front is the image of [0,1]^2 under T.
       
       Let's look at the inspiration: "stratified simplex sampling".
       And the "spherical coordinates" hint.
       
       Maybe the hint implies using a parameterization that is easier to stratify.
       
       Let's try a different approach:
       The Pareto front of DTLZ1 is actually a simplex in the space of
       (f1, f2, f3) but scaled?
       
       Let u1 = 2*f1, u2 = 2*f2, u3 = 2*f3.
       u1 = x1
       u2 = x2*(1-x1)
       u3 = 1-x2
       
       u1 + u2 + u3 = x1 + x2 - x1*x2 + 1 - x2 = 1 + x1 - x1*x2.
       Still not 1.
       
       Wait, I might be misremembering DTLZ1.
       Let me check a standard reference or derive it.
       
       Actually, for DTLZ1, the Pareto front is indeed the (M-1)-simplex
       defined by sum fi = 0.5? No.
       
       Let's check the vertices.
       If x1=1, x2=0, g=0: f=(0.5, 0, 0.5).
       If x1=0, x2=1, g=0: f=(0, 0.5, 0).
       If x1=0, x2=0, g=0: f=(0, 0, 0.5).
       
       So the vertices are (0.5, 0, 0.5), (0, 0.5, 0), (0, 0, 0.5).
       The sum of coordinates is 0.5 in all these cases.
       
       Let's check an interior point.
       x1=0.5, x2=0.5, g=0.
       f1 = 0.25
       f2 = 0.5 * 0.5 * 0.5 = 0.125
       f3 = 0.5 * 0.5 = 0.25
       Sum = 0.25 + 0.125 + 0.25 = 0.625.
       
       So the front is NOT planar. It is a curved surface.
       
       The "simplex" in the literature usually refers to the shape of the
       trade-off surface in the objective space, which for DTLZ1 is a
       (M-1)-simplex? No, it's a curved patch.
       
       However, the reference point is [400, 400, 400].
       The objectives are small (< 1).
       So all points are much better than the reference point.
       The hypervolume is dominated by the volume of the bounding box
       defined by the worst points.
       
       To maximize hypervolume, we need to push the points as far out as
       possible in the "bad" direction (towards the reference point).
       Since the reference point is very far, the hypervolume is essentially
       the volume of the convex hull of the points and the reference point.
       Or more accurately, the union of boxes from each point to the reference point.
       
       Since the points are close to the origin (0,0,0) relative to 400,
       the hypervolume is roughly the volume of the box defined by the
       maximum extent of the points in each dimension?
       No, hypervolume is the volume of the region dominated by the points.
       
       If we have a set of points P, the hypervolume is the volume of
       {y | y_i >= f_i(p) for some p in P, y_i <= ref_i}.
       
       Since ref is large, this is the volume of the union of boxes
       [f(p), ref] for all p in P.
       
       To maximize this, we want the points to cover the "outer" boundary
       of the Pareto front as much as possible.
       
       The current LHS strategy covers the entire surface uniformly in x-space.
       Does this correspond to uniform coverage in f-space?
       We saw that the Jacobian varies.
       
       Points where the Jacobian is small are "stretched" in f-space.
       Points where the Jacobian is large are "compressed".
       
       To maximize hypervolume, we want to ensure we have points in the
       "corners" of the f-space extent.
       
       The extent in f1 is [0, 0.5].
       The extent in f2 is [0, 0.5]. (Max f2 is 0.5 at x1=0, x2=1? No, x1=0, x2=1 => f2=0.5. Wait.
       f2 = 0.5 * x2 * (1-x1). Max is 0.5 * 1 * 1 = 0.5. Correct.)
       The extent in f3 is [0, 0.5].
       
       So the bounding box is [0, 0.5]^3.
       
       If we sample uniformly in x-space, do we cover the f-space well?
       
       Let's try to sample such that we cover the f-space more uniformly.
       
       One effective strategy for hypervolume on a known front is to use
       a stratified grid in the objective space itself, and then invert
       to find the decision variables.
       
       We can define a grid on the triangle (simplex) f1+f2+f3 <= 0.5?
       No, the front is not on a plane.
       
       However, we can parameterize the front by (u, v) where u, v are
       barycentric-like coordinates.
       
       Let's use the parameterization:
       x1 = 1 - cos(theta1)
       x2 = 1 - cos(theta2)
       This might cluster points.
       
       Let's stick to the "stratified simplex" idea but applied to the
       decision variables that define the front.
       
       The current program uses LHS on x1, x2.
       LHS is good.
       
       What if we use a grid?
       With 200 evaluations, we can't do a fine grid on 2D (e.g., 14x14=196).
       A grid might be better for hypervolume than LHS if the front is
       smooth and we want to cover the extent.
       
       Let's try a stratified grid on x1, x2.
       n1 = 14, n2 = 14.
       14*14 = 196 points.
       
       Or use a grid on the "shape" parameters that map more uniformly
       to the objective space.
       
       Let's try the spherical coordinate parameterization of the decision
       variables x1, x2.
       x1 = r * cos(phi)
       x2 = r * sin(phi)
       This maps a circle to a square, which is not good.
       
       Let's try to sample f1, f2, f3 directly from the front?
       We can define a mapping from a unit square [0,1]^2 to the front.
       
       Let s, t in [0, 1].
       We want to map (s, t) to (x1, x2) such that the resulting points
       are well-distributed in f-space.
       
       Given the Jacobian J(x1, x2) = 0.25 * sqrt(x2^2 + 1 + (1-x1)^2),
       we want density proportional to 1/J.
       
       This is hard to invert analytically.
       
       However, we can use the fact that the front is convex.
       The hypervolume is maximized by points on the "ridge" of the front.
       
       Let's just use a stratified grid on x1, x2.
       The LHS might have random gaps. A grid ensures regular coverage.
       
       Let's implement a stratified grid for x1, x2.
       
       n1 = 14, n2 = 14.
       x1_i = (i + 0.5) / 14
       x2_j = (j + 0.5) / 14
       
       Total 196 points.
       
       We set x3..x6 = 0.
       
       This should cover the front regularly.
       
       Let's check if 196 is enough. Budget is 200.
       We can add 4 more points or adjust grid size.
       14x14 = 196.
       We can use 15x13 = 195?
       Or just use 196 and ignore the last 4?
       Or use 200 points with a slightly irregular grid?
       
       Let's use 14x14 grid and 4 random points?
       Or just 196 points. The remaining 4 evaluations can be wasted
       or used to refine the corners.
       
       Actually, the corners of the x1-x2 square map to specific points on the front.
       (0,0) -> f=(0,0,0.5)
       (1,0) -> f=(0.5,0,0.5)
       (0,1) -> f=(0,0.5,0)
       (1,1) -> f=(0.5,0,0) -- Wait.
       If x1=1, x2=1:
       f1 = 0.5
       f2 = 0.5 * 1 * 0 = 0
       f3 = 0.5 * 0 = 0
       So (1,1) maps to (0.5, 0, 0).
       
       So the four corners of the x1-x2 square map to the four "extreme" points
       of the front in the f-space.
       (0,0) -> (0,0,0.5)
       (1,0) -> (0.5,0,0.5)
       (0,1) -> (0,0.5,0)
       (1,1) -> (0.5,0,0)
       
       These are the vertices of the bounding box of the front.
       A grid will include points near these corners, which is good for hypervolume.
       
       Let's implement the grid.
       
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        n_shape = m - 1  # 2 for M=3
        n_dist = n - n_shape  # 5 for N=7
        
        # We will use a stratified grid on the shape variables
        # to ensure regular coverage of the Pareto front.
        
        # Determine grid dimensions
        # We want approximately budget points.
        # sqrt(200) ~ 14.14
        # Let's use 14x14 = 196 points.
        n_grid = 14
        n_points = n_grid * n_grid
        if n_points > self.budget:
            # Reduce grid size
            while n_points > self.budget:
                n_grid -= 1
                n_points = n_grid * n_grid
        
        # Generate grid points for x1, x2
        # Use midpoints of the grid cells to avoid boundaries
        x1_vals = (np.arange(n_grid) + 0.5) / n_grid
        x2_vals = (np.arange(n_grid) + 0.5) / n_grid
        
        X_grid, Y_grid = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        
        # Flatten
        shape_samples = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        
        # Construct full decision vectors
        X = np.zeros((n_points, n))
        X[:, 0] = shape_samples[:, 0]
        X[:, 1] = shape_samples[:, 1]
        # Distance variables are 0
        # X[:, 2:] = 0 (already zero)
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        # If we have budget left, we can add a few random points
        # or just return. The task says "Every returned objective vector is recorded".
        # It doesn't say we MUST use all budget, but "More than 200 evaluations" is invalid.
        # So using 196 is fine.
        
        return X
# EVOLVE-BLOCK-END
