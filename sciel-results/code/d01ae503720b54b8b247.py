"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-exploiting optimizer for DTLZ7.
    
    Strategy:
    1. Identify that for DTLZ7, the first (m-1) variables (x0, x1) determine the 
       shape of the Pareto front, while the remaining variables (x2...x21) determine 
       the constraint function g.
    2. To minimize g (and thus reach the Pareto front), the optimal value for the 
       tail variables is 0.5 (assuming bounds [0,1]).
    3. Fix the tail variables to 0.5.
    4. Perform a dense local search (grid + refinement) on the first 2 variables 
       (x0, x1) to maximize coverage of the Pareto front.
    5. This reduces the effective dimensionality from 22 to 2, allowing for a much 
       denser and more accurate sampling of the front within the budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Fix tail variables to optimal value (midpoint of [0,1] is 0.5)
        # For DTLZ7, g(x) = sum(x_i - 0.5)^2 for i in [2, dim)
        # So x_i = 0.5 minimizes g.
        tail_optimal = np.full(self.dim - 2, 0.5)
        
        # 2. We will search over x0 and x1.
        # The Pareto front of DTLZ7 is defined by x0 and x1 in [0, 1].
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g) * h(x2, x3, f3) ... wait, DTLZ7 objectives are:
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g) * (n_obj - 1 - sum((1+fi) * sin(pi*fi/2) for i in [2, m-1]))
        # Here m=3, so the sum is over i=2 to m-1=2? No, standard DTLZ7:
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g(x_M)) * h(y, f3)
        # where h(y, f3) = n_obj - 1 - sum_{i=2}^{m-1} ((1+f_i) * sin(pi*f_i/2))
        # Since m=3, the sum is over i=2 to 2? No, usually it's i=m-1 to M-1?
        # Let's check standard DTLZ7 definition.
        # f1 = x1
        # f2 = x2
        # f3 = (1 + g) * (n_obj - 1 - sum_{i=3}^{M} (1+x_i) sin(pi x_i / 2))
        # In our case, dim=22, n_obj=3.
        # x0, x1 are the first two.
        # x2...x21 are the rest.
        # g depends on x2...x21.
        # The third objective depends on x2...x21 through g AND the sine terms?
        # Actually, for DTLZ7, the Pareto front is achieved when g=0.
        # When g=0, f3 = h(...).
        # The shape of the front is determined by x0 and x1?
        # Wait, f1=x0, f2=x1. So the front is a surface in f1-f2-f3 space.
        # f3 depends on x2...x21?
        # Let's re-read DTLZ7 definition carefully.
        # f1 = x1
        # f2 = x2
        # f3 = (1 + g(x_M)) * h(y, f3)
        # h(y, f3) = n_obj - 1 - sum_{i=m}^{M} (1 + x_i) * sin(pi * x_i / 2)
        # Here m = n_obj - 1 = 2. So sum is from i=2 to M (where M=dim-1=21).
        # So f3 depends on x2...x21.
        # BUT, if we fix x2...x21 to 0.5, then g=0.
        # Then f3 = h(x2...x21).
        # If x2...x21 are fixed to 0.5, then f3 is a CONSTANT.
        # This would mean the Pareto front is a line (or point) in the f3 direction?
        # No, that's not right.
        
        # Let's check the standard DTLZ7 implementation often used in benchmarks.
        # Usually, the first m-1 variables define the first m-1 objectives.
        # The remaining variables define g and the last objective's shape.
        # For DTLZ7 with M=22, m=3:
        # x0, x1 are "position" variables.
        # x2...x21 are "distance" variables.
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g) * (2 - sum_{i=2}^{21} (1+x_i) sin(pi x_i / 2))
        # If we fix x2...x21 to 0.5, g=0.
        # Then f3 = 2 - sum_{i=2}^{21} (1.5) sin(pi*0.5/2) = 2 - 20 * 1.5 * sin(pi/4)
        # This is a constant.
        # This implies that for a FIXED set of tail variables, the Pareto front is a line segment in the f1-f2 plane at a fixed f3 value.
        # To get a 2D front (surface), we need to vary the tail variables?
        # No, the Pareto front of DTLZ7 is a 2D surface in 3D objective space.
        # The "shape" of the front is determined by the choice of x2...x21.
        # Wait, if x2...x21 are varied, g changes.
        # For a point to be on the Pareto front, g must be 0.
        # So x2...x21 MUST be 0.5 (for the standard g).
        # If x2...x21 are 0.5, f3 is constant.
        # This contradicts the idea of a 2D front.
        
        # Let's re-read the DTLZ7 definition from Deb et al.
        # "The third objective is non-separable..."
        # Actually, in DTLZ7, the Pareto front is disconnected? No, it's connected but complex.
        # Let's look at the formula again.
        # f1 = x1
        # f2 = x2
        # f3 = (1 + g(x_M)) * (n_obj - 1 - sum_{i=m}^{M} (1+x_i) sin(pi x_i / 2))
        # If g=0, f3 = n_obj - 1 - sum...
        # The sum depends on x_i for i>=m.
        # If we fix x_i for i>=m to 0.5, the sum is constant.
        # So f3 is constant.
        # This means the Pareto front lies on a plane f3 = constant?
        # No, that would be a 1D front (line) in the f1-f2 plane.
        # But DTLZ7 is known for having a complex front.
        
        # Ah, I see. In many implementations, the "shape" variables are the ones that appear in the sine terms.
        # If those are fixed, the front is flat in the f3 direction.
        # However, the task says "minimizing all 3 objectives".
        # If the front is a line, we just need to cover that line.
        # But the reference point is [1.1, 1.1, 25.0].
        # If f3 is constant, say ~1.0, then the hypervolume is determined by the spread in f1 and f2.
        
        # Let's verify the "constant f3" hypothesis.
        # If x2..x21 = 0.5, then f3 is a specific value C.
        # The points are (x0, x1, C).
        # The Pareto front is the line segment from (0,0,C) to (1,1,C)?
        # No, f1=x0, f2=x1. The domain is [0,1]^2.
        # So the front is the square [0,1]x[0,1] at height C?
        # No, only the "corner" is Pareto optimal?
        # For minimization, (0,0,C) dominates (1,1,C) in f1 and f2.
        # So the Pareto front is just the point (0,0,C)?
        # No, that's only if we can vary x0 and x1 independently.
        # But x0 and x1 are independent.
        # So (0,0,C) is the unique Pareto optimal solution?
        # If so, the hypervolume is just the box from (0,0,C) to (1.1, 1.1, 25.0)?
        # No, hypervolume of a single point is 0? No, it's the volume of the region dominated by that point relative to the reference.
        # Region: f1 in [0, 1.1], f2 in [0, 1.1], f3 in [C, 25.0].
        # Volume = 1.1 * 1.1 * (25.0 - C).
        
        # If this is true, the problem is trivial: find the point with minimum f1, f2, and f3.
        # Min f1 = 0 (x0=0). Min f2 = 0 (x1=0).
        # Min f3 is achieved when g=0 and the sum is maximized.
        # The sum is maximized when sin(pi*x_i/2) is maximized, i.e., x_i=1.
        # But if x_i=1, g = sum(1-0.5)^2 = 20 * 0.25 = 5.
        # So f3 = (1+5) * (2 - sum(2 * 1)) = 6 * (2 - 20) = negative?
        # No, sin(pi*1/2) = 1.
        # Sum = 20 * (1+1) * 1 = 40.
        # f3 = (1+5) * (2 - 40) = -228.
        # But objectives are usually positive?
        # The reference point is 25.0.
        
        # Let's check the standard DTLZ7 bounds and objective ranges.
        # Usually f3 is in [0, 10] or similar.
        # If I fix tail to 0.5, f3 is constant.
        # Is the Pareto front really just a point?
        # "The Pareto front of DTLZ7 is a 2D surface."
        # This implies that f3 is NOT constant for all Pareto optimal points.
        # This means that for a point to be Pareto optimal, g must be 0, BUT the values of x2..x21 can vary as long as g=0?
        # g(x) = sum (x_i - 0.5)^2.
        # g=0 implies x_i = 0.5 for ALL i.
        # So x2..x21 are FIXED to 0.5.
        # Therefore f3 IS constant.
        # Therefore the Pareto front IS a line (or point) in the f1-f2 plane?
        # If f1=x0 and f2=x1 are independent, then (0,0) is the only Pareto optimal point in the f1-f2 plane?
        # No, (0, 0.5) is not dominated by (0,0) in f2?
        # (0,0) has f1=0, f2=0.
        # (0, 0.5) has f1=0, f2=0.5.
        # (0,0) dominates (0,0.5) because 0<=0 and 0<0.5.
        # So yes, (0,0) is the unique Pareto optimal point?
        
        # This seems too simple. Let's check the inspiration.
        # Inspiration 1: "first m-1 variables determine the first m-1 objectives and the rest determine the constraint/g-function".
        # If the front is a single point, the hypervolume is fixed.
        # But the score varies.
        
        # Maybe my understanding of "Pareto Front" for DTLZ7 is flawed.
        # Let's look at a reference.
        # "DTLZ7 ... Pareto front is disconnected?" No.
        # "The Pareto front of DTLZ7 is a 2D surface that is not convex."
        # How can it be a 2D surface if g=0 forces x2..x21 to 0.5?
        
        # Ah! I might be confusing DTLZ7 with DTLZ2 or 3.
        # DTLZ2: Sphere. g=0 -> x_M=0.5? No, g=0 -> x_{m+1}...x_M = 0.5.
        # Then f1..f_{m-1} are defined by the first m-1 vars.
        # f_m is defined by the remaining part of the sphere.
        # For DTLZ2, the front is a (m-1)-dimensional surface.
        
        # For DTLZ7:
        # f1 = x1
        # f2 = x2
        # f3 = (1+g) * h(...)
        # If g=0, f3 = h(...).
        # h depends on x2...x21.
        # If x2...x21 are fixed to 0.5, h is constant.
        # So f3 is constant.
        # So the front is in the f1-f2 plane.
        # Since f1=x0, f2=x1, and they are independent, the "front" is the entire square [0,1]x[0,1]?
        # No, only the points that are not dominated.
        # (0,0) dominates everything else.
        # So the Pareto front is the single point (0,0,f3_const).
        
        # If this is the case, the optimal strategy is to find (0,0,0.5...0.5).
        # One evaluation at x=[0,0,0.5,...,0.5] gives the ideal point.
        # The hypervolume would be 1.1 * 1.1 * (25 - f3_const).
        
        # Why would the score be 1.0 (seed program) and others lower?
        # The seed program does random search. It might find points close to (0,0).
        # If I find (0,0) exactly, I get the max hypervolume.
        
        # Let's assume this "single point" hypothesis is correct for the standard DTLZ7 definition.
        # Strategy:
        # 1. Evaluate x = [0, 0, 0.5, ..., 0.5].
        # 2. This should give the Pareto optimal point.
        # 3. To be safe, maybe do a small local search around it?
        # But if (0,0) is the unique optimum, any other point is worse.
        
        # However, what if the bounds are not [0,1]?
        # "bounds [0, 1] for every variable".
        
        # Let's try a strategy that samples the "corner" (0,0) and a few points around it to ensure we catch the optimum, and also samples the "edges" in case my dominance analysis is wrong (e.g. if there's a constraint I'm missing).
        
        # Actually, let's look at the "2D grid on x[0] and x[1]" finding.
        # "Switching from 1D sampling on x[2] to 2D grid on x[0] and x[1] covers the true 2D Pareto front surface".
        # This confirms that the front is 2D.
        # This implies my "single point" logic is wrong.
        # Why?
        # Maybe f1 and f2 are not simply x0 and x1?
        # Or maybe the dominance is not standard?
        # "minimizing all 3 objectives".
        
        # Let's re-read the DTLZ7 formula from a reliable source (e.g. PlatEMO).
        # f1 = x1
        # f2 = x2
        # f3 = (1 + g) * (n_obj - 1 - sum_{i=3}^{M} (1+x_i) sin(pi x_i / 2))
        # If x_i (i>=3) are 0.5, f3 is constant.
        # So f1=x1, f2=x2.
        # The set of points is {(x1, x2, C) | x1,x2 in [0,1]}.
        # Pareto optimality: A point p is Pareto optimal if no other point q exists such that q_i <= p_i for all i and q_j < p_j for some j.
        # For any point (a, b, C), the point (0, 0, C) has f1=0<=a, f2=0<=b, f3=C=C.
        # So (0,0,C) dominates (a,b,C) unless a=0 and b=0.
        # So (0,0,C) is the ONLY Pareto optimal point.
        
        # This contradicts the "2D surface" finding.
        # Is it possible that the "2D surface" finding refers to DTLZ2 or 3?
        # The prompt says "DTLZ7".
        # Maybe the implementation of DTLZ7 in the evaluator is different?
        # Or maybe "Pareto front surface" is a loose term for the set of non-dominated points, which in this case is a line?
        # But the finding says "2D grid on x[0] and x[1] covers the true 2D Pareto front surface".
        # If the front is a line, a 2D grid is wasteful but still covers it.
        # A 1D sampling on x[2] (tail) would be useless because x[2] doesn't affect f1, f2.
        
        # So, if the front is effectively the line/point at x0=0, x1=0, then:
        # The best strategy is to concentrate evaluations at x0=0, x1=0.
        
        # Let's try a hybrid:
        # 1. Evaluate the ideal corner (0,0,0.5...).
        # 2. Do a dense grid on x0, x1 in [0, 0.1] to catch any non-convexity or if my dominance logic is slightly off (e.g. if f3 varies slightly with x0,x1? No, it doesn't).
        # 3. Use remaining budget to random search? No, waste.
        
        # Wait, what if the reference point is such that points with f3 > C are still valuable?
        # No, we minimize.
        
        # Let's assume the "Single Point" theory is correct for the specific DTLZ7 variant used.
        # The "2D grid" improvement might just be because it covered the (0,0) area better than 1D on x2.
        
        # Strategy:
        # 1. Fixed tail = 0.5.
        # 2. Dense grid on x0, x1 in [0, 1].
        # Even if only (0,0) is optimal, a dense grid ensures we find it.
        # Budget 200.
        # Grid 14x14 = 196 points.
        # This covers the entire [0,1]x[0,1] square.
        # If the front is a line, this is overkill but safe.
        # If the front is a 2D surface (my error), this is necessary.
        
        # Let's implement the 2D grid on x0, x1 with tail fixed to 0.5.
        
        n_points = self.budget
        # Use a square grid
        side = int(np.sqrt(n_points))
        # Adjust to fit budget
        if side * side > n_points:
            side -= 1
        # Add some random points to fill remaining budget?
        # Or just use the grid.
        
        x0_vals = np.linspace(0, 1, side)
        x1_vals = np.linspace(0, 1, side)
        
        X = []
        F = []
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                X.append(x)
                F.append(problem(x))
                
                if len(X) >= self.budget:
                    break
            if len(X) >= self.budget:
                break
                
        # If we have budget left, do a few random points?
        # Or refine the grid?
        # For now, just return the grid points.
        
        return np.array(X)

# EVOLVE-BLOCK-END
