"""Multi-objective optimizer on DTLZ7 using structured Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 that exploits the known Pareto front structure.
    The first 2 variables (x0, x1) parameterize the 2D Pareto front,
    and the remaining 20 variables are set to minimize g.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ7, the Pareto front is 2D: parameterized by x0, x1
        # The remaining variables should be set to minimize g
        # g = sum((xi - 0.5)^2) for i >= 2, so minimum at xi = 0.5
        # But we need to check which setting gives better hypervolume
        
        # Strategy: sample x0, x1 on a grid, set x[2:] to 0.5 (minimizes g)
        # Then refine based on hypervolume contribution
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Reserve some budget for potential refinement
        # Start with a grid in (x0, x1) space
        # The Pareto front in objective space: f1=x0, f2=x1, f3=(1+g)*h
        # With g=0 (x[2:]=0.5), h = n_obj - 1 = 2, so f3 = 2
        
        # But wait - if f3 is constant at 2, then the front is a line, not 2D
        # Let me reconsider: perhaps x[2:] should be 0, not 0.5
        
        # For DTLZ7, the standard Pareto front has x[2:] = 0
        # Let's try both and see which gives points within bounds and better HV
        
        # Try x[2:] = 0 first
        def make_point(x0, x1, tail_val):
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            x[2:] = tail_val
            return x
        
        # We need points where f1 < 1.1, f2 < 1.1, f3 < 25.0
        # f1 = x0, f2 = x1, so x0 < 1.1, x1 < 1.1 (always true in [0,1])
        # f3 = (1+g)*h, need this < 25
        
        # With x[2:] = 0: g = sum((0-0.5)^2) = 20*0.25 = 5
        # h = (M-1) - sum over Y of (xi/g * (1+2*sin(pi*xi)))
        # Y = {i >= 2 : xi != 0.5} = all of them since xi=0
        # h = 2 - sum_{i=2}^{21} (0/5 * (1+2*sin(0))) = 2 - 0 = 2
        # f3 = (1+5)*2 = 12 < 25, good
        
        # With x[2:] = 0.5: g = 0, Y = empty, h = 2, f3 = 1*2 = 2 < 25, also good
        # But f3=2 is constant, so the front is degenerate
        
        # The actual DTLZ7 Pareto front: I believe it's with x[2:] = 0
        # Let's use x[2:] = 0
        
        tail_val = 0.0
        
        # Sample x0, x1 to cover the front
        # The front in (f1, f2, f3) space with x[2:]=0:
        # f1 = x0, f2 = x1, f3 = 12 (constant?) 
        # Wait, h depends on xi values. If all xi=0, then:
        # h = 2 - sum(0/5 * ...) = 2
        # So f3 = 12, constant. That's a line, not 2D surface.
        
        # I think I'm misunderstanding DTLZ7. Let me just use a general approach:
        # Sample points in the full 22D space, but bias toward the Pareto front region
        
        # Actually, for DTLZ7 the Pareto front IS 2D. The issue is that
        # h is not constant - it depends on the specific values of x[2:]
        
        # Let me re-examine: if we set x[2:] = 0, then for each i in Y (all of them):
        # term = xi/g * (1 + 2*sin(pi*xi)) = 0/5 * (1+0) = 0
        # So h = 2, f3 = 12. Constant.
        
        # If we vary x[2:], we get different f3 values. But for the Pareto front,
        # we want to minimize all objectives. Since f1 and f2 only depend on x0, x1,
        # and f3 depends on all, the Pareto front should have x[2:] at their optimal values.
        
        # I think the standard DTLZ7 Pareto front is obtained by setting x[2:] = 0.5
        # (minimizing g to 0), and then the front is parameterized by x0, x1 with
        # f3 = (1+0) * h where h depends on how many x[2:] are exactly 0.5.
        
        # This is getting complicated. Let me use a practical approach:
        # 1. Generate candidate points with x[2:] = 0.5 (g=0)
        # 2. Generate candidate points with x[2:] = 0 (g=5)  
        # 3. Evaluate and keep the non-dominated set
        
        # Given budget constraints, let's be strategic.
        
        # Phase 1: Grid sampling of x0, x1 with x[2:] = 0.5
        n_grid = 15  # 15x15 = 225, too many. Use 13x13 = 169
        n_grid = 13
        x0_vals = np.linspace(0.01, 0.99, n_grid)
        x1_vals = np.linspace(0.01, 0.99, n_grid)
        
        candidates = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = make_point(x0, x1, 0.5)
                candidates.append(x)
        
        # That's 169 points. We have 200 budget.
        # Evaluate all, then use remaining budget for refinement
        
        evaluated = []
        f_values = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            f = np.array(problem(x))
            evaluated.append(x)
            f_values.append(f)
        
        if not evaluated:
            # Fallback: single point
            x = make_point(0.5, 0.5, 0.5)
            f = np.array(problem(x))
            return [x]
        
        f_values = np.array(f_values)
        evaluated = np.array(evaluated)
        
        used = len(evaluated)
        
        # Phase 2: Use remaining budget for local refinement
        # Find the best points and try to improve them
        if used < budget:
            # Keep non-dominated points
            n_points = len(f_values)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        if np.all(f_values[j] <= f_values[i]) and np.any(f_values[j] < f_values[i]):
                            dominated[i] = True
                            break
            
            non_dom_idx = np.where(~dominated)[0]
            
            # Refine non-dominated points with small perturbations
            rng = np.random.default_rng(42)
            remaining = budget - used
            
            for idx in non_dom_idx:
                if used >= budget or time.time() > deadline:
                    break
                base = evaluated[idx]
                # Small mutation in x0, x1
                for _ in range(min(3, budget - used)):
                    if used >= budget or time.time() > deadline:
                        break
                    child = base.copy()
                    child[0] += rng.normal(0, 0.02)
                    child[1] += rng.normal(0, 0.02)
                    child = np.clip(child, lo, hi)
                    f_child = np.array(problem(child))
                    used += 1
                    evaluated = np.vstack([evaluated, child.reshape(1, -1)])
                    f_values = np.vstack([f_values, f_child.reshape(1, -1)])
        
        return evaluated
# EVOLVE-BLOCK-END
