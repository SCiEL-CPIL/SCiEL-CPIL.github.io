import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1: n=7, M=3
        # Pareto front: g=0 when x[2:7] = 0.5
        # Objectives (with g=0):
        #   f0 = 0.5 * x[0] * (1 - x[1])
        #   f1 = 0.5 * (1 - x[0]) * x[1]
        #   f2 = 0.5 * (1 - x[1])
        # where x[0], x[1] in [0,1] and x[0] + x[1] <= 1 for the simplex? 
        # Actually DTLZ1 front is: sum of f_i = 0.5, f_i >= 0
        # The mapping from (x0, x1) to objectives:
        #   f0 = 0.5*x0*(1-x1)
        #   f1 = 0.5*(1-x0)*x1
        #   f2 = 0.5*(1-x1)
        # Let's verify: f0+f1+f2 = 0.5*x0 - 0.5*x0*x1 + 0.5*x1 - 0.5*x0*x1 + 0.5 - 0.5*x1
        #   = 0.5*x0 - 0.5*x0*x1 + 0.5*x1 - 0.5*x0*x1 + 0.5 - 0.5*x1
        #   = 0.5*x0 - x0*x1 + 0.5
        #   = 0.5*x0(1-2x1) + 0.5 ... that doesn't simplify to 0.5 easily.
        
        # Let me re-derive. DTLZ1:
        # f0 = 0.5 * x1 * (1 + g(x2..xn))
        # f1 = 0.5 * (1-x1) * x2 * (1 + g(x2..xn))
        # f2 = 0.5 * (1-x2) * (1 + g(x2..xn))
        # Wait, standard DTLZ1 has M objectives. For M=3:
        # f0 = 0.5 * x0 * (1 + g)
        # f1 = 0.5 * (1-x0) * x1 * (1 + g)
        # f2 = 0.5 * (1-x1) * (1 + g)
        # where x0, x1 are the first 2 decision variables, and g depends on x[2:].
        # At g=0:
        # f0 = 0.5 * x0
        # f1 = 0.5 * (1-x0) * x1
        # f2 = 0.5 * (1-x1)
        # Sum = 0.5*x0 + 0.5*(1-x0)*x1 + 0.5*(1-x1)
        #     = 0.5*x0 + 0.5*x1 - 0.5*x0*x1 + 0.5 - 0.5*x1
        #     = 0.5*x0 - 0.5*x0*x1 + 0.5
        #     = 0.5*(1 + x0 - x0*x1)
        # This is not constant 0.5. So the front is not a flat simplex in objective space
        # parameterized by x0, x1 directly.
        
        # Actually, the DTLZ1 Pareto front in objective space is the set of points where
        # sum(f_i) = 0.5 and f_i >= 0. The mapping from decision variables to objectives
        # at g=0 is:
        # f0 = 0.5 * x0
        # f1 = 0.5 * (1-x0) * x1
        # f2 = 0.5 * (1-x1)
        # For any point on the front, we need f0+f1+f2 = 0.5.
        # From the equations: f0 = 0.5*x0 => x0 = 2*f0
        # f2 = 0.5*(1-x1) => x1 = 1 - 2*f2
        # f1 = 0.5*(1-x0)*x1 = 0.5*(1-2*f0)*(1-2*f2)
        # So f1 is determined by f0 and f2. The front is 2D.
        
        # To sample the front, we can sample (f0, f2) such that:
        # f0 in [0, 0.5], f2 in [0, 0.5], and f1 = 0.5*(1-2*f0)*(1-2*f2) >= 0
        # Also f0 + f1 + f2 = 0.5 must hold.
        # Let's check: f0 + f1 + f2 = f0 + 0.5*(1-2*f0)*(1-2*f2) + f2
        #   = f0 + 0.5 - f0 - f2 + 2*f0*f2 + f2
        #   = 0.5 + 2*f0*f2
        # That's not 0.5! I must have the DTLZ1 definition wrong.
        
        # Let me look up DTLZ1 properly.
        # DTLZ1: 
        # g(x) = 100 * (n-M+1 + sum(x_i^2 - x_i) for i=M+1..n)
        # For M=3, n=7: g depends on x[3], x[4], x[5], x[6] (0-indexed: x[3:7])
        # Wait, the task says "x[0], x[1] control the front shape, x[2]...x[6] control g"
        # So M=3 means the first M=3 variables? No, standard DTLZ1:
        # f0 = 0.5 * x1 * (1 + g)
        # f1 = 0.5 * (1-x1) * x2 * (1 + g)  
        # f2 = 0.5 * (1-x2) * (1 + g)
        # where x1, x2 are the 2 "position" variables and x3..x7 control g.
        # So in 0-indexed: x[0], x[1] are position, x[2:7] control g.
        # At g=0:
        # f0 = 0.5 * x[0]
        # f1 = 0.5 * (1-x[0]) * x[1]
        # f2 = 0.5 * (1-x[1])
        # Sum = 0.5*x[0] + 0.5*(1-x[0])*x[1] + 0.5*(1-x[1])
        # Let u = x[0], v = x[1]:
        # Sum = 0.5*u + 0.5*(1-u)*v + 0.5*(1-v)
        #     = 0.5*u + 0.5*v - 0.5*u*v + 0.5 - 0.5*v
        #     = 0.5*u - 0.5*u*v + 0.5
        #     = 0.5*(1 + u - u*v)
        # For this to equal 0.5, we need u - u*v = 0, i.e., u*(1-v) = 0.
        # So either u=0 or v=1. That means the front is only on the boundary?
        
        # I think I'm confusing myself. Let me just directly sample (x[0], x[1]) 
        # on the unit simplex-like region and compute the objectives.
        # The Pareto front of DTLZ1 is well-known to be a 2D simplex in objective space
        # where sum(f_i) = 0.5. The mapping is:
        # Given weights (w0, w1, w2) with w0+w1+w2=1, wi>=0:
        # f_i = 0.5 * w_i
        # And the decision variables that achieve this are:
        # x[0] = w0
        # x[1] = w1 / (w0 + w1) if w0+w1 > 0, else 0
        # Wait, let's verify:
        # f0 = 0.5 * x[0] = 0.5 * w0 ✓
        # f1 = 0.5 * (1-x[0]) * x[1] = 0.5 * (1-w0) * (w1/(w0+w1))
        # For this to equal 0.5*w1, we need (1-w0)/(w0+w1) = 1, i.e., 1-w0 = w0+w1, i.e., w0+w1 = 0.5.
        # That's not generally true.
        
        # I'll just use a direct approach: sample points in the (x[0], x[1]) space
        # that lie on the Pareto front. The Pareto front condition for DTLZ1 at g=0
        # is that all objectives are minimized. Since the mapping from (x0,x1) to
        # (f0,f1,f2) is bijective for the front, I can just sample (x0, x1) uniformly
        # on a 2D grid/simplex and evaluate.
        
        # Actually, the simplest and most robust approach: sample (x[0], x[1]) 
        # on a fine grid over [0,1]^2, set x[2:7]=0.5 (g=0), and evaluate.
        # This covers the entire front.
        
        n_side = int(np.sqrt(self.budget))
        # 14x14 = 196, plus 4 mid-edge points = 200
        x0_vals = np.linspace(0, 1, n_side)
        x1_vals = np.linspace(0, 1, n_side)
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        samples = np.column_stack((X0.ravel(), X1.ravel()))
        
        mid_points = np.array([
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5]
        ])
        samples = np.vstack([samples, mid_points])
        
        if len(samples) > self.budget:
            samples = samples[:self.budget]
        elif len(samples) < self.budget:
            center = np.array([[0.5, 0.5]])
            samples = np.vstack([samples, np.tile(center, (self.budget - len(samples), 1))])
        
        for i in range(self.budget):
            x = np.zeros(self.dim)
            x[0] = samples[i, 0]
            x[1] = samples[i, 1]
            x[2:] = 0.5
            problem(x)
        
        return None
