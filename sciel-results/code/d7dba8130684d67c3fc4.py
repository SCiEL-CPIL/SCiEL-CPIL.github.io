"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using adaptive variable identification.
    
    Strategy:
    1. Screen variables to identify 'position' variables (affecting direction of f) 
       and 'g' variables (affecting distance from origin).
    2. Optimize 'g' variables to their minimum (typically 0 for DTLZ4 when x=0.5).
    3. Sample 'position' variables on a dense grid to cover the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # --- Phase 1: Variable Importance Screening ---
        # We need to determine which indices are 'position' and which are 'g'.
        # In DTLZ4, g is minimized at x=0.5 (for standard bounds [0,1]).
        # Position variables determine the direction.
        
        # Hypothesis: 
        # Standard DTLZ4: x[0..k-1] are position, x[k..n-1] are g.
        # Usually k = m-1 = 2 for m=3. But let's verify.
        
        # We will test a few candidate splits.
        # For each candidate split (k), we assume x[0:k] are pos, x[k:] are g.
        # We evaluate a point where g-variables are at 0.5 (min g) and pos-variables are varied.
        # If the resulting points are close to the ideal front (small norm), it's likely correct.
        
        # However, we don't know the "ideal front" norm a priori without evaluating.
        # But we know that for DTLZ4, min g = 0. So the points should be on the unit sphere (scaled by 0.5? or 1?).
        # Standard DTLZ4: f_i = (1+g) * h_i. If g=0, f is on the unit sphere.
        
        # Let's try to find the split k that minimizes the variance of the norm of f 
        # when we fix g-vars to 0.5 and vary pos-vars.
        # Actually, if the split is wrong, some "pos" vars will actually be g-vars.
        # Varying a g-var away from 0.5 increases g, thus increasing norm.
        # So we want to find the set of variables that can be varied while keeping norm low.
        
        # A simpler heuristic for DTLZ4 specifically:
        # The g-function in DTLZ4 is typically: g(x) = 100 * (n_g + sum((x_i - 0.5)^10 - 10*x_i*(x_i-1)^2))
        # This is minimized at x_i = 0.5.
        # The position variables are usually the first m-1 variables.
        
        # Let's test k=1, k=2, k=3, k=4.
        # For each k, we construct a point:
        # x[0:k] = some random values (to test if they are pos)
        # x[k:] = 0.5 (to minimize g)
        # And another point:
        # x[0:k] = 0.5
        # x[k:] = 0.5
        # If k is correct, the first point should have norm ~1 (if on unit sphere) and the second should also have norm ~1.
        # If k is too small (e.g., k=1 but actually k=2), then x[1] is a g-var.
        # In the first point, x[1] is random (not 0.5), so g > 0, norm > 1.
        # In the second point, x[1]=0.5, so g=0, norm ~1.
        
        # So, for a candidate k, we evaluate:
        # 1. A point with x[k:] = 0.5 and x[0:k] random.
        # 2. A point with x[:] = 0.5.
        # If the norms are similar, k is likely correct (or larger).
        # If norm 1 >> norm 2, then some x in [0:k] are actually g-vars (should be 0.5).
        
        # Let's do a quick screening.
        # We have budget 200. Let's use 10 evaluations for screening.
        
        candidates_k = [1, 2, 3, 4, 5]
        scores_k = {}
        
        # Base point: all 0.5
        x_base = np.array([0.5 * (lo[i] + hi[i]) for i in range(self.dim)])
        f_base = problem(x_base)
        norm_base = np.linalg.norm(f_base)
        
        # For each k, test if varying x[0:k] changes the norm significantly.
        # If it does, those variables are likely g-vars (because they affect magnitude).
        # If it doesn't (or changes direction but not magnitude much), they are pos-vars.
        
        # Wait, pos-vars change direction, so f vector changes, but norm stays ~1 (if g=0).
        # G-vars change magnitude.
        
        # So for a candidate k:
        # Generate a point x_test where x[k:] = 0.5, and x[0:k] are perturbed from 0.5.
        # If x[0:k] are pos-vars, norm(x_test) should be close to norm_base.
        # If any x in [0:k] are g-vars, norm(x_test) should be > norm_base.
        
        rng = np.random.RandomState(42)
        
        for k in candidates_k:
            if k >= self.dim:
                continue
            x_test = np.array([0.5 * (lo[i] + hi[i]) for i in range(self.dim)])
            # Perturb x[0:k]
            for i in range(k):
                # Perturb significantly, e.g., to 0.1 or 0.9
                x_test[i] = lo[i] + 0.8 * (hi[i] - lo[i])
            
            f_test = problem(x_test)
            norm_test = np.linalg.norm(f_test)
            
            # Ratio of norms
            ratio = norm_test / (norm_base + 1e-9)
            scores_k[k] = ratio
            
        # We want the smallest k such that ratio is close to 1.
        # If k=1 is pos, and k=2 is g, then:
        # k=1: x[0] perturbed (pos), x[1:] 0.5. Norm ~ 1. Ratio ~ 1.
        # k=2: x[0] perturbed (pos), x[1] perturbed (g), x[2:] 0.5. Norm > 1. Ratio > 1.
        # So we want the smallest k where ratio is NOT close to 1? 
        # No, we want the set of pos-vars to be exactly those that don't increase norm.
        # So we want the largest k such that all x in [0:k] are pos-vars?
        # Actually, if k is too large, we include g-vars in the "pos" set, causing high norm.
        # If k is too small, we exclude pos-vars from the "pos" set? No, we just don't perturb them.
        
        # Let's refine:
        # We want to identify the true number of pos-vars, let's call it M.
        # For a candidate k:
        # If k <= M: We perturb only pos-vars. Norm stays low. Ratio ~ 1.
        # If k > M: We perturb pos-vars AND some g-vars. Norm increases. Ratio > 1.
        
        # So we want the largest k such that ratio is close to 1.
        
        best_k = 1
        threshold = 1.1 # Allow 10% increase
        
        for k in candidates_k:
            if k not in scores_k:
                continue
            if scores_k[k] < threshold:
                best_k = k
            else:
                break # Assuming k is sorted, once it fails, larger k will likely fail too
        
        # In case all pass (unlikely), or none pass (weird), fallback to 2.
        if best_k not in scores_k or scores_k[best_k] > threshold:
            best_k = 2 # Most common for 3 obj
        
        # --- Phase 2: Optimize G-variables ---
        # We assume x[best_k:] are g-vars.
        # We want to minimize g. For DTLZ4, min is at 0.5.
        # But let's do a few local evaluations to be sure.
        # We have budget left.
        
        # Let's use a small local search around 0.5 for the g-vars.
        # Or just trust 0.5 if the screening was successful.
        # Given the time constraint, trusting 0.5 is safer and faster.
        
        g_vars = np.arange(best_k, self.dim)
        pos_vars = np.arange(0, best_k)
        
        # --- Phase 3: Sample Position Variables ---
        # We have budget - (1 + len(candidates_k)) evaluations used.
        # Let's say we used ~6.
        # Remaining: ~194.
        
        # We need to sample x[0:best_k] to cover the Pareto front.
        # For M=2, the front is a quarter circle.
        # For M=3, it's a quarter sphere.
        
        # We will use a grid or quasi-random points.
        # A grid is good for hypervolume.
        
        n_evals_used = 1 + len(candidates_k)
        remaining_budget = self.budget - n_evals_used
        
        if remaining_budget <= 0:
             return x_base
        
        # Generate points for pos-vars
        # If best_k == 1: 1D line
        # If best_k == 2: 2D square -> circle
        # If best_k == 3: 3D cube -> sphere
        
        # We want to distribute points uniformly on the surface of the unit hyper-sphere (orthant).
        # For M=2: x0 = cos(theta), x1 = sin(theta). theta in [0, pi/2].
        # For M=3: x0 = cos(phi)cos(theta), x1 = cos(phi)sin(theta), x2 = sin(phi).
        
        # However, the problem maps x in [0,1] to angles.
        # Standard DTLZ4 mapping:
        # h_i(x) is defined using x directly.
        # Usually x is in [0,1] and mapped to [0, pi/2] or used directly in trig.
        
        # Let's just sample x[0:best_k] uniformly in [0,1] on a grid.
        # This ensures coverage of the variable space, which maps to the front.
        
        # Number of points per dimension for grid:
        # If best_k=2, n_per_dim = sqrt(remaining_budget)
        # If best_k=3, n_per_dim = cbrt(remaining_budget)
        
        if best_k == 1:
            n_per_dim = remaining_budget
        elif best_k == 2:
            n_per_dim = int(np.sqrt(remaining_budget))
        elif best_k == 3:
            n_per_dim = int(np.cbrt(remaining_budget))
        else:
            # Fallback to random if too many dims
            n_per_dim = int(np.power(remaining_budget, 1.0/best_k))
            
        n_per_dim = max(2, n_per_dim)
        
        # Generate grid points
        # We need to be careful not to exceed budget.
        
        # For simplicity and robustness, let's use a Halton sequence or just a dense grid.
        # Grid is better for HV.
        
        # Create grid
        if best_k == 1:
            pos_vals = np.linspace(0, 1, n_per_dim)
            Pos = pos_vals[:, np.newaxis]
        elif best_k == 2:
            v1 = np.linspace(0, 1, n_per_dim)
            v2 = np.linspace(0, 1, n_per_dim)
            V1, V2 = np.meshgrid(v1, v2)
            Pos = np.column_stack((V1.flatten(), V2.flatten()))
        elif best_k == 3:
            v1 = np.linspace(0, 1, n_per_dim)
            v2 = np.linspace(0, 1, n_per_dim)
            v3 = np.linspace(0, 1, n_per_dim)
            V1, V2, V3 = np.meshgrid(v1, v2, v3)
            Pos = np.column_stack((V1.flatten(), V2.flatten(), V3.flatten()))
        else:
            # Random sampling for high dims
            Pos = rng.rand(remaining_budget, best_k)
        
        # Check if we exceed budget
        if len(Pos) > remaining_budget:
             # Sample a subset
             indices = rng.choice(len(Pos), remaining_budget, replace=False)
             Pos = Pos[indices]
        
        # Construct full population
        Pop = np.zeros((len(Pos), self.dim))
        
        # Set pos vars
        Pop[:, :best_k] = Pos
        
        # Set g vars to 0.5
        for i in range(best_k, self.dim):
            Pop[:, i] = 0.5 * (lo[i] + hi[i])
        
        # Evaluate
        # We should check time
        if time.time() > deadline:
            return x_base
            
        F = np.array([problem(x) for x in Pop])
        
        return Pop
# EVOLVE-BLOCK-END
