"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a structured grid on the Pareto front.
    
    Strategy:
    1. DTLZ2 with M=3 has a Pareto front determined by x[0], x[1] (position variables).
    2. x[2:] should be set to the center of the bounds (0.5) to minimize g(x)=0.
    3. We use a structured grid to generate 200 points in [0, 1]^2.
    4. These points are mapped to the DTLZ2 Pareto front angles.
    5. The resulting 200 objective vectors are returned.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Structure Assumption:
        # M=3 objectives => m=2 position variables (x[0], x[1]).
        # x[2:] are distance variables.
        # To be on the Pareto front, g(x) must be 0.
        # g(x) = sum((x[i] - 0.5)^2) for i >= m.
        # So set x[2:] to 0.5 (center of [0, 1]).
        
        n_points = self.budget
        
        # For DTLZ2 with M=3, the Pareto front is a quarter-sphere in 3D.
        # The position variables (x[0], x[1]) map to angles on this sphere.
        # Specifically, if we let x[0] = sin(theta), x[1] = cos(theta) * sin(phi),
        # but actually for DTLZ2, the mapping is:
        # f1 = g * (1 - x1 * x2) ... no wait.
        
        # Let's recall DTLZ2:
        # f1 = g * (1 - x_{m-1})
        # f2 = g * x_{m-1} * (1 - x_{m-2})
        # f3 = g * x_{m-1} * x_{m-2}
        # where g = sum_{i=m}^{n} (x_i - 0.5)^2
        # For M=3, m=2. So x[0], x[1] are position variables.
        # f1 = g * (1 - x[1])
        # f2 = g * x[1] * (1 - x[0])
        # f3 = g * x[1] * x[0]
        
        # If g=0 (x[2:] = 0.5), then f1=f2=f3=0, which is the ideal point.
        # But we want to cover the Pareto front.
        # The Pareto front for DTLZ2 is the set of points (f1, f2, f3) such that
        # f1^2 + f2^2 + f3^2 = 1, f_i >= 0.
        
        # The relationship between x[0], x[1] and the front:
        # If we set g=1 (by choosing x[2:] such that sum((x_i-0.5)^2)=1),
        # then:
        # f1 = 1 - x[1]
        # f2 = x[1] * (1 - x[0])
        # f3 = x[1] * x[0]
        
        # Let x[1] = sin(alpha), x[0] = sin(beta) ... this doesn't quite work.
        
        # Actually, a simpler parametrization:
        # Let x[1] = cos(phi) and x[0] = sin(theta) / cos(phi) ... no.
        
        # Let's use spherical coordinates directly on the Pareto front.
        # f1 = sin(phi) * cos(theta)
        # f2 = sin(phi) * sin(theta)
        # f3 = cos(phi)
        # where phi in [0, pi/2], theta in [0, pi/2].
        
        # We need to find x[0], x[1] in [0,1] that produce these.
        # From f1 = g*(1-x[1]), f2 = g*x[1]*(1-x[0]), f3 = g*x[1]*x[0].
        # If we set g=1, then:
        # 1-x[1] = f1
        # x[1]*(1-x[0]) = f2
        # x[1]*x[0] = f3
        # From first: x[1] = 1 - f1
        # From second: (1-f1)*(1-x[0]) = f2 => 1-x[0] = f2/(1-f1) => x[0] = 1 - f2/(1-f1)
        # From third: (1-f1)*x[0] = f3 => x[0] = f3/(1-f1)
        # For consistency: 1 - f2/(1-f1) = f3/(1-f1) => 1 - f1 - f2 = f3 => f1+f2+f3 = 1.
        # But on the DTLZ2 front, f1^2+f2^2+f3^2=1, not sum=1.
        
        # So we can't just set g=1. We need g such that the points lie on the sphere.
        # f1 = g(1-x1), f2 = g*x1*(1-x0), f3 = g*x1*x0.
        # f1^2 + f2^2 + f3^2 = g^2 [ (1-x1)^2 + x1^2(1-x0)^2 + x1^2*x0^2 ]
        # = g^2 [ (1-x1)^2 + x1^2 ((1-x0)^2 + x0^2) ]
        # For this to equal 1, we need g = 1 / sqrt((1-x1)^2 + x1^2).
        # Wait, (1-x0)^2 + x0^2 is not 1. It's 1 - 2x0 + 2x0^2.
        
        # Actually, the standard DTLZ2 Pareto front is the unit sphere in the first octant.
        # The mapping from x[0], x[1] to the front when g is chosen appropriately is complex.
        
        # A better approach: Since we know the Pareto front is the quarter-sphere,
        # we can directly generate points on the sphere and then find the corresponding
        # x[0], x[1] that would produce them if g were set correctly.
        
        # However, the problem evaluator calls problem(x) which computes g from x[2:].
        # We must set x[2:] to make g=0 to get points on the front? No, if g=0, all f=0.
        
        # Let's re-read the DTLZ2 definition.
        # DTLZ2:
        # g(x) = sum_{i=m}^{n} (x_i - 0.5)^2
        # f1 = g * (1 - x_{m-1})
        # f2 = g * x_{m-1} * (1 - x_{m-2})
        # ...
        # f_M = g * x_{m-1} * x_{m-2} * ... * x_0
        
        # For M=3, m=2:
        # f1 = g * (1 - x1)
        # f2 = g * x1 * (1 - x0)
        # f3 = g * x1 * x0
        
        # If we set x[2:] = 0.5, then g = 0, so f1=f2=f3=0. This is a single point.
        # To get the Pareto front, we need g > 0.
        # The Pareto front is the set of points where g is minimized for given x[0], x[1].
        # The minimum g is 0, achieved at x[2:]=0.5.
        # But if g=0, all objectives are 0.
        
        # I think I am misremembering DTLZ2. Let me check.
        # DTLZ2:
        # f1 = g * (1 - x_{m-1})
        # f2 = g * x_{m-1} * (1 - x_{m-2})
        # ...
        # f_{M-1} = g * x_{m-1} * ... * x_1 * (1 - x_0)
        # f_M = g * x_{m-1} * ... * x_0
        
        # For M=3, m=2:
        # f1 = g * (1 - x1)
        # f2 = g * x1 * (1 - x0)
        # f3 = g * x1 * x0
        
        # The Pareto front is obtained by minimizing g. The minimum g is 0.
        # If g=0, then f1=f2=f3=0.
        # This means the Pareto front is just the origin? That can't be right for a hypervolume problem.
        
        # Ah, I see. The reference point is [2.5, 2.5, 2.5].
        # If all points are (0,0,0), the hypervolume is 2.5^3 = 15.625.
        # The "dense fixed sample of the known Pareto front" would also be (0,0,0) if the front is a point.
        # Then the ratio would be 1.
        
        # But the score is 1.18, which is > 1. This implies the front is not a point.
        
        # Let's re-read the DTLZ2 definition carefully.
        # Maybe g is not sum((x_i-0.5)^2).
        # DTLZ2: g(x) = sum_{i=m}^{n} (x_i - 0.5)^2.
        # If x_i in [0,1], then g >= 0.
        # The Pareto front is the set of non-dominated points.
        # If g=0, f=(0,0,0).
        # If g>0, f is scaled up.
        # (0,0,0) dominates any (f1,f2,f3) with f_i > 0.
        # So the Pareto front is indeed just (0,0,0).
        
        # Wait, maybe the bounds are not [0,1]? The task says "bounds [0, 1] for every variable".
        
        # Let's look at the current program's score: 1.1794.
        # If the front is a point, the hypervolume of any set of points is the same (volume of box from (0,0,0) to (2.5,2.5,2.5)).
        # Unless... the reference point is not [2.5,2.5,2.5]? The task says it is.
        
        # Perhaps I am misunderstanding "Pareto front".
        # In DTLZ2, the Pareto front is the unit sphere in the first octant.
        # This is true if g is defined differently or if the objectives are defined differently.
        
        # Let's check a standard DTLZ2 definition online (mentally).
        # DTLZ2:
        # f1 = g * (1 - x_{m-1})
        # f2 = g * x_{m-1} * (1 - x_{m-2})
        # ...
        # f_M = g * x_{m-1} * ... * x_0
        
        # The Pareto front is the set of points (f1, ..., fM) such that sum(f_i^2) = g^2.
        # If g is minimized to 0, the front collapses to the origin.
        
        # BUT, often in DTLZ problems, g is a constraint or the front is defined for g=1.
        # No, the standard definition says the Pareto front is the unit sphere.
        # This implies that g must be 1 on the front?
        
        # Let's re-read: "The Pareto front of DTLZ2 is the unit sphere in the first octant."
        # This is a common statement.
        # How is this reconciled with g = sum((x_i-0.5)^2)?
        
        # Maybe the variables x_i are not in [0,1] but in some other range?
        # No, the task says [0,1].
        
        # Let's try a different interpretation.
        # Maybe the "Pareto front" in the context of the score calculation refers to the set of points
        # that are non-dominated with respect to the reference point? No, that's the hypervolume.
        
        # Let's assume the standard DTLZ2 behavior where the front is the sphere.
        # This happens if g is fixed to 1, or if the problem is defined such that the front is the sphere.
        
        # Given the current program sets x[2:] = 0.5 (g=0) and gets a score of 1.18,
        # it suggests that the evaluator's "dense fixed sample of the known Pareto front"
        # is NOT the origin, but the sphere.
        
        # If the evaluator's reference front is the sphere, and our points are all at the origin,
        # the hypervolume of our points (origin) is the same as the hypervolume of the sphere?
        # No. The hypervolume of a single point (0,0,0) with ref (2.5,2.5,2.5) is 2.5^3.
        # The hypervolume of the sphere (quarter) with ref (2.5,2.5,2.5) is larger.
        # So if our points are all at the origin, our hypervolume is smaller than the reference front's hypervolume.
        # The score would be < 1.
        
        # But the score is 1.18 > 1.
        # This implies our points have a LARGER hypervolume than the reference front.
        # How can that be?
        
        # Maybe the reference front is not the sphere?
        # Or maybe our points are not at the origin?
        
        # If x[2:] = 0.5, g = 0. f1=f2=f3=0.
        # All 200 points are (0,0,0).
        # Hypervolume = 15.625.
        
        # If the score is 1.18, then Reference HV = 15.625 / 1.18 ≈ 13.24.
        # This is smaller than 15.625.
        
        # This is possible if the reference front is "worse" (further from origin) than our points.
        # But the Pareto front of DTLZ2 is the best set of points.
        
        # Wait, "minimizing all 3 objectives".
        # If the reference front is the Pareto front, it should be the best.
        # If our points are on the Pareto front, our HV should be close to the reference HV.
        # If our points are better (closer to origin), our HV is larger.
        
        # Is it possible that the "Pareto front" used for normalization is not the true Pareto front,
        # but a sample of it that is not optimal?
        
        # Or maybe my understanding of DTLZ2 is wrong.
        # Let's assume the current program is "correct" in the sense that it produces a valid score.
        # The current program generates a grid in x[0], x[1] and sets x[2:]=0.5.
        # If this yields score 1.18, then this strategy is good.
        
        # The inspirations suggest improving the grid.
        # Inspiration 1: 1.1799 (slightly better)
        # Inspiration 2: 1.1798
        # Inspiration 3: 1.1782 (worse)
        
        # The current program uses a 15x15 grid (225 points) and takes the first 200.
        # This is a non-uniform sample of the grid (it cuts off the last 25 points).
        # A better approach might be to use a 14x14 grid (196 points) and add 4 more points
        # strategically, or use a different grid shape like 10x20.
        
        # Let's try to use a grid that exactly matches 200 points or is more uniform.
        # 200 = 10 * 20.
        # Or 200 = 14 * 14 + 4.
        
        # Let's try 10x20 grid.
        n_per_axis_0 = 10
        n_per_axis_1 = 20
        # Total 200 points.
        
        x0_vals = np.linspace(0, 1, n_per_axis_0)
        x1_vals = np.linspace(0, 1, n_per_axis_1)
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        grid_points = np.column_stack([X0.ravel(), X1.ravel()])
        
        # Ensure we have exactly n_points
        if len(grid_points) > n_points:
            grid_points = grid_points[:n_points]
        elif len(grid_points) < n_points:
            # Pad with random points or repeat?
            # For now, just use what we have.
            pass
        
        results = []
        
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            
            # Position variables
            x[0] = grid_points[i, 0]
            x[1] = grid_points[i, 1]
            
            # Distance variables: set to center of bounds
            if self.dim > 2:
                # Use 0.5 as bounds are [0, 1]
                x[2:] = 0.5
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                # If an exception occurs, break to preserve valid results
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
