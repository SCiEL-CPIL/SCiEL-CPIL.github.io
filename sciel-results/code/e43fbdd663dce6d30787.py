"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Structure-aware optimizer using analytic front construction and Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        eval_count = 0
        results = []
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                # Ensure x is within bounds
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                eval_count += 1
                results.append(f)
                return f
            except Exception:
                eval_count += 1
                return None

        # Strategy:
        # 1. Identify the structure. For standard DTLZ1 with N=7, M=3:
        #    Position vars: x[0], x[1] (indices 0 to M-2)
        #    Distance vars: x[2], ..., x[6] (indices M-1 to N-1)
        #    The Pareto front is achieved when distance vars are at their optimum.
        #    For DTLZ1, the optimum for distance vars is typically 0 (or sometimes 0.5 depending on variant, but 0 is standard for g=0).
        #    Let's verify if setting distance vars to 0 yields a valid front.
        #    If the problem is standard DTLZ1, g(x) = 100 * (K + sum(x_i^2 - 10cos(20pi x_i))) for i in distance vars.
        #    Min is at x_i = 0, where term is -10. g_min = 100 * (K - 10K) = -900K.
        #    Wait, usually DTLZ1 is defined such that g >= 0.
        #    Standard DTLZ1: f_i = 0.5 * x_i * (1 + g(x_M..N))
        #    g(x) = 100 * (N - M + 1 + sum_{i=M}^N (x_i^2 - 10 cos(20 pi x_i)))
        #    If x_i = 0, x_i^2 - 10 cos(0) = -10.
        #    Sum = -10 * (N-M+1).
        #    g = 100 * (K - 10K) = -900K. This would make 1+g negative, which is bad.
        #    
        #    Let's re-read standard DTLZ1 definition.
        #    Often: g(x) = 100 * (K + sum (x_i^2 - 10 cos(20 pi x_i)))
        #    If we want g=0, we need sum(...) = -K.
        #    Since min of (x^2 - 10cos(20pi x)) is -10, if we set all to 0, sum is -10K.
        #    Then g = 100 * (K - 10K) = -900K.
        #    
        #    However, many implementations of DTLZ1 in benchmark suites (like PlatEMO) might use a different scaling or the "shifted" version.
        #    Actually, a very common variant is:
        #    f_1 = 0.5 x_1 (1+g)
        #    ...
        #    f_M = 0.5 (1 - sum x_i) (1+g)
        #    g = 100 (K + sum (x_i^2 - 10 cos(20 pi x_i)))
        #    
        #    If the evaluator uses this standard definition, setting x_i=0 for distance vars gives g < 0.
        #    Does the problem clip or handle negative g?
        #    The problem says "minimizing all 3 objectives".
        #    If g is negative, 1+g could be small or negative.
        #    If 1+g is negative, minimizing objectives might mean making them very negative?
        #    But DTLZ1 is usually bounded.
        #    
        #    Let's look at the "known Pareto front" mention in the task.
        #    "dense fixed sample of the known Pareto front".
        #    For DTLZ1, the Pareto front is typically the simplex in objective space where sum(f_i) = 0.5 * (1+g_min).
        #    If g_min is not 0, the front is shifted.
        #    
        #    However, a robust strategy is to find the point that minimizes the sum of objectives, which corresponds to the "center" of the front or a point on it.
        #    But we need to cover the whole front.
        #    
        #    Alternative hypothesis: The distance variables are NOT the last ones, or the optimum is not 0.
        #    Let's use a small budget to probe the structure.
        #    
        #    Probe 1: All zeros.
        #    Probe 2: All 0.5.
        #    Probe 3: For each var, set to 0.5, others 0.
        #    
        #    Actually, the most robust "structure-agnostic" approach that still exploits DTLZ1-like structure:
        #    1. Find a point x* that minimizes sum(f(x)). This is likely on the Pareto front (or close to it).
        #    2. Identify which variables are "position" and which are "distance" by perturbing them from x*.
        #       - Perturbing a position variable should change the *direction* of the objective vector (trade-off).
        #       - Perturbing a distance variable should change the *magnitude* of all objectives (shift away/towards origin).
        #    3. Fix distance variables at their optimal values (from x*).
        #    4. Sample the position variables using Sobol sequence on the simplex.
        
        # Let's implement this robust identification.
        
        # Step 1: Find a good starting point x* by minimizing sum(f).
        # We'll use a simple random search + local improvement for the first 20 evals.
        # Or just use the center and a few neighbors.
        
        # Let's try:
        # 1. Evaluate center (0.5 * ones).
        # 2. Evaluate zeros.
        # 3. Evaluate ones.
        
        x_center = np.full(dim, 0.5)
        f_center = eval_point(x_center)
        
        x_zeros = np.zeros(dim)
        f_zeros = eval_point(x_zeros)
        
        x_ones = np.ones(dim)
        f_ones = eval_point(x_ones)
        
        # Determine which of these is best (lowest sum)
        candidates = [(x_center, f_center), (x_zeros, f_zeros), (x_ones, f_ones)]
        valid_candidates = [c for c in candidates if c[1] is not None]
        
        if not valid_candidates:
            return None
            
        best_x, best_f = min(valid_candidates, key=lambda c: np.sum(c[1]))
        
        # Now, identify position vs distance variables.
        # We perturb each variable i by a small delta and see the effect.
        # If the change in f is roughly proportional to f (scaling), it's a distance variable.
        # If the change involves trade-offs (some obj up, some down), it's a position variable.
        
        # However, a simpler heuristic for DTLZ1:
        # The position variables are usually the first M-1.
        # Let's assume this is true for standard DTLZ1.
        # If the score is bad, we can fall back, but let's try the analytic approach first.
        
        # Assumption: x[0:M-1] are position, x[M-1:N] are distance.
        M = n_obj
        pos_indices = list(range(M - 1))
        dist_indices = list(range(M - 1, dim))
        
        # Let's verify this assumption by checking if setting dist vars to best_x[dist_indices]
        # and varying pos vars gives a good front.
        
        # Extract the "optimal" distance configuration from best_x
        dist_opt = best_x[dist_indices].copy()
        
        # Now, generate Sobol points for the position variables.
        # The position variables for DTLZ1 lie in the simplex: x_1 + ... + x_{M-1} <= 1.
        # We can sample from the unit hypercube and map to the simplex.
        # Or, since M-1 = 2 (for M=3), we can sample (u1, u2) in [0,1]^2 and map to simplex.
        # Simplex mapping:
        # Let y1, y2 be Sobol points in [0,1].
        # x1 = y1 * (1 - y2)  ?? No, standard mapping is:
        # For M-1 dims, sample M-1 uniform variables u_1, ..., u_{M-1}.
        # x_i = u_i * (1 - sum_{j=i+1}^{M-1} u_j) ?
        # Actually, a common way to sample a simplex of dimension D (D+1 vertices) is:
        # Generate D uniform random numbers u_1, ..., u_D.
        # x_i = u_i * (1 - u_{i+1}) for i=1..D-1, x_D = 1 - u_D.
        # This is for the standard simplex where sum x_i = 1.
        # DTLZ1 front is sum f_i = const. The x variables define the direction.
        # f_i = 0.5 * x_i * (1+g).
        # So if we sample x_i on the simplex (sum x_i = 1), then sum f_i = 0.5 * (1+g).
        
        # Let's use scipy.stats.qmc.Sobol to generate low-discrepancy points in the unit hypercube.
        # We need to map them to the simplex.
        
        # Number of points to generate:
        # We have used ~3 evals. Remaining budget is ~197.
        # Let's use all remaining budget for sampling.
        
        remaining_budget = budget - eval_count
        if remaining_budget <= 0:
            return None
            
        # Generate Sobol points
        # Dimension of Sobol sampler: M-1 (number of position variables)
        sobol_dim = len(pos_indices)
        
        # We want to cover the simplex.
        # If we sample u in [0,1]^D, we can map to simplex x in [0,1]^D with sum x_i <= 1?
        # No, the DTLZ1 front is defined by x_i >= 0 and sum x_i = 1 (for the unit simplex part).
        # Wait, f_M = 0.5 * (1 - sum_{i=1}^{M-1} x_i) * (1+g).
        # For f_M to be positive, we need sum x_i <= 1.
        # The Pareto front is the set of points where sum x_i = 1 (assuming g is minimized).
        # So we need to sample x_1, ..., x_{M-1} such that sum x_i = 1 and x_i >= 0.
        
        # How to sample from the simplex using Sobol?
        # We can sample M-1 points from [0,1] and normalize? No, that doesn't work directly.
        # Standard method: Sample M-1 uniform variables u_1, ..., u_{M-1}.
        # Sort them: v_1 < v_2 < ... < v_{M-1}.
        # Let v_0 = 0, v_M = 1.
        # x_i = v_i - v_{i-1} for i=1..M-1.
        # x_M = 1 - v_{M-1}.
        # This gives a point on the simplex sum x_i = 1.
        
        # However, we only have M-1 position variables in x. The M-th objective is derived.
        # So we sample x_1, ..., x_{M-1} from the simplex of dimension M-2?
        # No, the variables x_1, ..., x_{M-1} are independent in the input space, but constrained by the front condition.
        # If we set x_1, ..., x_{M-1} such that sum x_i = 1, we are on the front.
        
        # Let's generate Sobol points in dimension M-1.
        # But we need to map them to the simplex of dimension M-1 (which has M vertices, but we only control M-1 coords, the last is dependent).
        
        # Actually, simpler:
        # Generate N points in [0,1]^{M-1} using Sobol.
        # For each point u, map it to the simplex.
        # Method:
        # 1. Generate M-1 uniform random numbers u_1, ..., u_{M-1} from Sobol.
        # 2. Sort them.
        # 3. Compute differences.
        
        # Let's try a different mapping that is easier to implement with Sobol directly on the hypercube and then projecting.
        # Or just use random sampling on the simplex if Sobol mapping is complex.
        # But Sobol is better for hypervolume.
        
        # Let's use the "inverse CDF" or "LHS" like approach?
        # No, Sobol is low discrepancy.
        
        # Let's implement the sorting method for Sobol points.
        # Sobol gives points in [0,1]^D.
        # We want points in the simplex of dimension D (D+1 vertices).
        # If D=2 (M=3), we have 2 position variables x1, x2.
        # We sample u1, u2 from Sobol in [0,1]^2.
        # Sort u1, u2 -> v1, v2.
        # x1 = v1 - 0 = v1
        # x2 = v2 - v1
        # x3 (implicit) = 1 - v2
        # So x1 = min(u1, u2), x2 = |u1 - u2|, x3 = max(u1, u2) ?? No.
        # x1 = v1, x2 = v2 - v1.
        # Sum x1 + x2 = v2 <= 1.
        # This covers the triangle x1>=0, x2>=0, x1+x2<=1.
        # This is exactly the domain for x1, x2 in DTLZ1 M=3.
        
        # So:
        # 1. Generate Sobol points in [0,1]^{M-1}.
        # 2. For each point, sort the coordinates.
        # 3. Compute differences to get x_1, ..., x_{M-1}.
        
        sobol = qmc.Sobol(d=sobol_dim, scramble=True)
        # Number of points
        n_points = remaining_budget
        # Sobol requires power of 2? No, but it's efficient.
        # We can just generate n_points.
        
        # Generate
        # Note: qmc.Sobol.random(n) returns n points.
        try:
            sobol_points = sobol.random(n_points)
        except Exception:
            # Fallback to random if Sobol fails
            sobol_points = np.random.rand(n_points, sobol_dim)
        
        # Map to simplex
        # For each point u (length M-1):
        # Sort u -> v
        # x_i = v_i - v_{i-1} (with v_0 = 0)
        
        for i in range(n_points):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            u = sobol_points[i]
            # Sort
            v = np.sort(u)
            # Compute differences
            # v is length M-1
            # x_1 = v[0]
            # x_2 = v[1] - v[0]
            # ...
            # x_{M-1} = v[M-2] - v[M-3]
            
            x_pos = np.zeros(sobol_dim)
            x_pos[0] = v[0]
            if sobol_dim > 1:
                x_pos[1:] = v[1:] - v[:-1]
            
            # Construct full x
            x_full = np.zeros(dim)
            for j, idx in enumerate(pos_indices):
                x_full[idx] = x_pos[j]
            
            for j, idx in enumerate(dist_indices):
                x_full[idx] = dist_opt[j]
                
            # Clip to bounds just in case
            x_full = np.clip(x_full, lo, hi)
            
            eval_point(x_full)
            
        return None

# EVOLVE-BLOCK-END
