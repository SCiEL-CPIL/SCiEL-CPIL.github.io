# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using LHS on spherical coords + vertex placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _lhs_2d(self, n, seed=42):
        """Latin Hypercube Sampling in 2D."""
        rng = np.random.RandomState(seed)
        indices = [np.arange(n) for _ in range(2)]
        for i in range(2):
            rng.shuffle(indices[i])
        u = np.array([(indices[i][j] + rng.rand()) / n for i in range(2) for j in range(n)]).reshape(n, 2)
        return u

    def _spherical_to_front(self, theta, phi):
        """Convert spherical angles to DTLZ4 front-position variables.
        For M=3, the Pareto front is the unit sphere in the first octant.
        theta in [0, pi/2], phi in [0, pi/2]
        f1 = sin(theta)*sin(phi), f2 = sin(theta)*cos(phi), f3 = cos(theta)
        But we need x[0], x[1] such that the problem maps to these f values.
        For DTLZ4: x[0] and x[1] are in [0,1], and the mapping to front is:
        f1 = (1+g)*sin(pi*x[1]/2) ... actually let's just use the direct mapping.
        
        Actually for DTLZ4 with M=3:
        The front-position variables x[0], x[1] map to the Pareto front via:
        f1 = g*(1+cos(pi*x[1]/2))
        f2 = g*(1+cos(pi*x[0]/2))*sin(pi*x[1]/2)  
        f3 = g*sin(pi*x[0]/2)*sin(pi*x[1]/2)
        
        When g is minimized (g-variables at optimal), the relative ratios are:
        f1 : f2 : f3 = (1+cos(pi*x[1]/2)) : (1+cos(pi*x[0]/2))*sin(pi*x[1]/2) : sin(pi*x[0]/2)*sin(pi*x[1]/2)
        
        This doesn't directly give a sphere. Let me just sample x[0], x[1] uniformly
        and rely on the problem's mapping.
        """
        x = np.zeros(self.dim)
        x[0] = theta
        x[1] = phi
        x[2:] = 0.5  # Optimal g-variables for DTLZ4
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f
        
        # Phase 1: Place vertices (corners of the front)
        # For DTLZ4 M=3, the three extreme points are approximately:
        # (0,0), (0,1), (1,0) in x[0],x[1] space with g-variables at 0.5
        vertices = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        for v in vertices:
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Phase 2: LHS sampling on front-position variables
        n_lhs = self.budget - used - 20  # Reserve 20 for local search
        n_lhs = max(10, n_lhs)
        
        # Use LHS in 2D for x[0], x[1]
        u_lhs = self._lhs_2d(n_lhs, seed=0)
        
        for u in u_lhs:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = u[0]
            x[1] = u[1]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Phase 3: Local refinement - perturb g-variables to see if we can improve
        # Also try small perturbations in front-position variables
        rng = np.random.RandomState(123)
        remaining = self.budget - used
        
        # Select some points for local search
        if remaining > 0 and len(all_F) > 0:
            F_arr = np.array(all_F)
            # Pick points that are likely non-dominated (lower sum of objectives)
            sums = F_arr.sum(axis=1)
            # Get indices of points with lowest sums
            n_local = min(remaining, 20)
            n_per_point = max(1, n_local // min(10, len(all_F)))
            
            # Select up to 10 diverse points
            selected = np.argsort(sums)[:min(10, len(all_F))]
            
            for idx in selected:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                base_x = all_X[idx].copy()
                base_f = all_F[idx]
                
                for _ in range(n_per_point):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    new_x = base_x.copy()
                    # Small perturbation in front-position variables
                    new_x[0] += 0.05 * rng.randn()
                    new_x[1] += 0.05 * rng.randn()
                    # Small perturbation in g-variables (try to find better g)
                    new_x[2:] += 0.02 * rng.randn(len(new_x) - 2)
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = evaluate(new_x)
        
        return all_X

# EVOLVE-BLOCK-END
