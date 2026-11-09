"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    Uses structural knowledge of DTLZ4 to initialize points on the Pareto front.
    Maintains a non-dominated set and uses remaining budget for local/global search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 300  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Non-dominated set
        nd_points = []  # List of (x, f)
        used_evals = 0
        
        def is_dominated(f_new, nd_f_list):
            for f_old in nd_f_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def is_dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_point(x, f):
            nonlocal nd_points, used_evals
            # Check if f is dominated by existing
            if is_dominated(f, [p[1] for p in nd_points]):
                return False
            
            # Remove points dominated by new
            new_nd = []
            for px, pf in nd_points:
                if not is_dominates(f, pf):
                    new_nd.append((px, pf))
            
            new_nd.append((x, f))
            nd_points = new_nd
            return True

        # 1. Initialization
        # We assume DTLZ4 structure:
        # First n_obj variables define the shape of the front (sphere).
        # Remaining variables should be 0.5 to minimize g(x).
        
        n_init = min(100, self.budget // 2)
        
        # Generate structured points on the assumed front
        # For DTLZ4, the front is defined by the first n_obj variables lying on the unit sphere in the positive orthant.
        # f_i = (1+g) * x_i ... (simplified: if g=0, f is proportional to x on the sphere)
        # Actually, for DTLZ2/4, f_i = (1+g) * x_i is NOT correct for all i.
        # Standard DTLZ: f_i = (1+g) * x_i for i=1..M is only true if the definition is f_i = (1+g)x_i and sum x_i^2 = 1.
        # Let's generate points on the unit sphere in the first n_obj dimensions.
        
        structured_xs = []
        for _ in range(n_init // 2):
            # Random direction in n_obj dims
            v = np.random.randn(self.n_obj)
            v = np.abs(v) # Ensure positive
            norm = np.linalg.norm(v)
            if norm == 0:
                v[0] = 1.0
                norm = 1.0
            v = v / norm
            
            x = np.zeros(self.dim)
            x[:self.n_obj] = v
            x[self.n_obj:] = 0.5 # Set to center to minimize g
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            structured_xs.append(x)
            
        # Generate random points
        random_xs = []
        for _ in range(n_init - len(structured_xs)):
            x = lo + (hi - lo) * np.random.rand(self.dim)
            random_xs.append(x)
            
        all_init_xs = structured_xs + random_xs
        
        # Evaluate initial points
        for x in all_init_xs:
            if time.time() > deadline:
                break
            f = np.array(problem(x))
            used_evals += 1
            add_point(x, f)
            
        # 2. Search Phase
        # Use remaining budget for local search around best points and global random search
        while used_evals < self.budget:
            if time.time() > deadline:
                break
                
            if not nd_points:
                # Fallback to random
                x = lo + (hi - lo) * np.random.rand(self.dim)
            else:
                # Mix of local and global
                if np.random.rand() < 0.7:
                    # Local search: mutate a random point from the non-dominated set
                    parent_x, parent_f = nd_points[np.random.randint(len(nd_points))]
                    # Adaptive mutation step
                    step_size = 0.1
                    x = parent_x + step_size * np.random.randn(self.dim)
                    x = np.clip(x, lo, hi)
                else:
                    # Global random
                    x = lo + (hi - lo) * np.random.rand(self.dim)
            
            f = np.array(problem(x))
            used_evals += 1
            add_point(x, f)
            
            # Optional: Prune if set gets too large to keep computation fast
            # Hypervolume calculation is expensive, but we only store non-dominated.
            # For 200 evals, the set size will be manageable.
            
        return nd_points
# EVOLVE-BLOCK-END
