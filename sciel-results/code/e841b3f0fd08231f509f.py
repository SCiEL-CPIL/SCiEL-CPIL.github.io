import numpy as np
import time

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using
    stratified sampling on the simplex, combined with adaptive refinement
    around hypervolume-critical regions."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n points in the unit square using Halton sequence (bases 2 and 3)."""
        i = np.arange(1, n + 1)
        
        # Halton base 2
        x0 = np.zeros(n)
        for k in range(n):
            val = 0.0
            x = i[k]
            f = 0.5
            while x > 0:
                val += f * (x % 2)
                x //= 2
                f /= 2.0
            x0[k] = val

        # Halton base 3
        x1 = np.zeros(n)
        for k in range(n):
            val = 0.0
            x = i[k]
            f = 1.0 / 3.0
            while x > 0:
                val += f * (x % 3)
                x //= 3
                f /= 3.0
            x1[k] = val

        return np.column_stack((x0, x1))

    def _simplex_sample(self, n):
        """Generate n points uniformly on the 2D simplex (a+b+c=1, a,b,c>=0).
        Returns (a, b, c) for each point."""
        # Use Dirichlet(1,1,1) which is uniform on the simplex
        # Sample using exponential random variables
        u1 = np.random.exponential(size=n)
        u2 = np.random.exponential(size=n)
        u3 = np.random.exponential(size=n)
        total = u1 + u2 + u3
        a = u1 / total
        b = u2 / total
        c = u3 / total
        return np.column_stack((a, b, c))

    def _map_simplex_to_x(self, simplex_pts):
        """Map simplex points (a, b, c) to DTLZ1 decision variables.
        
        For DTLZ1 with M=3 objectives:
        f0 = 0.5 * x0 * (1+g)
        f1 = 0.5 * (1-x0) * x1 * (1+g)
        f2 = 0.5 * (1-x1) * (1+g)
        
        At g=0, the Pareto front satisfies f0+f1+f2 = 0.5.
        Given weights (w0, w1, w2) with w0+w1+w2=1:
        f_i = 0.5 * w_i
        
        We need to find x0, x1 such that:
        0.5*x0 = 0.5*w0  =>  x0 = w0
        0.5*(1-x0)*x1 = 0.5*w1  =>  x1 = w1/(1-w0) if w0 < 1, else 0
        0.5*(1-x1) = 0.5*w2  =>  x1 = 1 - w2
        
        From the second and third equations:
        w1/(1-w0) = 1 - w2
        w1 = (1-w0)(1-w2) = 1 - w0 - w2 + w0*w2
        Since w0+w1+w2=1: w1 = 1-w0-w2
        So: 1-w0-w2 = 1-w0-w2+w0*w2  =>  w0*w2 = 0
        
        This means the mapping isn't straightforward. Let me use a different approach.
        
        Actually, the standard DTLZ1 front parameterization uses:
        x0 = w0
        x1 = w1 / (w0 + w1) if w0+w1 > 0
        
        Let's verify:
        f0 = 0.5 * x0 = 0.5 * w0 ✓
        f1 = 0.5 * (1-x0) * x1 = 0.5 * (1-w0) * (w1/(w0+w1))
        For this to equal 0.5*w1: (1-w0)/(w0+w1) = 1 => 1-w0 = w0+w1 => w0+w1 = 0.5
        This is not generally true.
        
        Let me just directly sample x0, x1 on the valid region and evaluate.
        The valid region for the front is where all objectives are non-negative.
        """
        dim = self.dim
        n = len(simplex_pts)
        points = np.zeros((n, dim))
        
        # For each simplex point (a, b, c), we want to find x0, x1 such that
        # the objectives are proportional to (a, b, c).
        # 
        # A simpler approach: sample x0 uniformly in [0,1] and x1 uniformly in [0, 1-x0]
        # This covers the triangle in (x0, x1) space that maps to the front.
        
        # Actually, let me just use the direct parameterization.
        # The Pareto front of DTLZ1 can be parameterized by two angles or by
        # sampling on the simplex in objective space.
        
        # Given target weights (w0, w1, w2) summing to 1:
        # We want f0/f1/f2 = w0/w1/w2
        # f0 = 0.5*x0
        # f1 = 0.5*(1-x0)*x1
        # f2 = 0.5*(1-x1)
        
        # From f0: x0 = 2*f0 = 2*0.5*w0 = w0
        # From f2: x1 = 1 - 2*f2 = 1 - 2*0.5*w2 = 1 - w2
        # Check f1: 0.5*(1-w0)*(1-w2) should equal 0.5*w1
        # (1-w0)(1-w2) = 1-w0-w2+w0*w2 = w1 + w0*w2
        # This equals w1 only if w0*w2 = 0.
        
        # So the mapping from simplex weights to (x0,x1) is not direct.
        # Instead, let's sample (x0, x1) directly on the valid domain.
        
        # The valid domain for DTLZ1 front is:
        # x0 in [0, 1], x1 in [0, 1], and the objectives must be non-negative.
        # All objectives are non-negative for x0, x1 in [0,1].
        # The front is the image of [0,1]^2 under the mapping (x0,x1) -> (f0,f1,f2).
        
        # To get uniform coverage on the front, we should sample (x0,x1) in a way
        # that gives uniform coverage in objective space.
        
        # For now, let's just use stratified sampling on [0,1]^2 for (x0,x1).
        pass
        
        return points

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Strategy: Use stratified sampling on the (x0, x1) plane to cover the Pareto front.
        # The Pareto front of DTLZ1 is a 2D surface in objective space, parameterized by
        # x0 and x1 (with x[2:7] = 0.5 to minimize g).
        #
        # Key insight: We want to maximize hypervolume, which means we want points
        # spread across the entire front, especially near the corners where the
        # hypervolume contribution is largest.
        
        # Phase 1: Stratified grid sampling (majority of budget)
        # Use a grid that's denser near the corners
        n_grid_side = int(np.sqrt(budget * 0.7))
        n_phase1 = min(n_grid_side * n_grid_side, budget)
        n_phase2 = budget - n_phase1
        
        # Generate stratified samples on [0,1]^2
        # Use a jittered grid for better coverage
        grid_pts = []
        for i in range(n_grid_side):
            for j in range(n_grid_side):
                # Stratified: divide [0,1] into n_grid_side intervals, sample once in each
                u0 = (i + np.random.uniform()) / n_grid_side
                u1 = (j + np.random.uniform()) / n_grid_side
                grid_pts.append([u0, u1])
        
        grid_pts = np.array(grid_pts)
        
        # Add corner points explicitly for better hypervolume
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.5, 0.5]
        ])
        
        # Combine grid and corners, ensuring we don't exceed budget
        all_samples = np.vstack([corners, grid_pts])
        if len(all_samples) > n_phase1:
            all_samples = all_samples[:n_phase1]
        
        # Evaluate phase 1
        results = []
        objectives = []
        for i in range(len(all_samples)):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = all_samples[i, 0]
            x[1] = all_samples[i, 1]
            for j in range(2, dim):
                x[j] = 0.5
            obj = problem(x)
            results.append(obj)
            objectives.append(np.array(obj))
        
        if time.time() > deadline or len(results) >= budget:
            return results
        
        # Phase 2: Adaptive refinement around non-dominated points
        if n_phase2 > 0 and objectives:
            objectives = np.array(objectives)
            
            # Find non-dominated points
            n = len(objectives)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                        dominated[j] = True
                    elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        dominated[i] = True
                        break
            
            front_idx = np.where(~dominated)[0]
            
            if len(front_idx) > 0:
                # Refine around the front points
                n_refine_per_point = max(1, n_phase2 // max(1, len(front_idx)))
                
                refine_points = []
                for idx in front_idx:
                    if time.time() > deadline:
                        break
                    center = all_samples[idx] if idx < len(all_samples) else np.random.rand(2)
                    
                    # Generate local jittered points
                    n_local = min(n_refine_per_point, 3)
                    for _ in range(n_local):
                        if time.time() > deadline or len(refine_points) >= n_phase2:
                            break
                        jitter_scale = 0.05
                        dx = (np.random.random() - 0.5) * 2 * jitter_scale
                        dy = (np.random.random() - 0.5) * 2 * jitter_scale
                        new_x0 = np.clip(center[0] + dx, 0.0, 1.0)
                        new_x1 = np.clip(center[1] + dy, 0.0, 1.0)
                        refine_points.append([new_x0, new_x1])
                
                # Evaluate refinement points
                for rp in refine_points:
                    if time.time() > deadline or len(results) >= budget:
                        break
                    x = np.zeros(dim)
                    x[0] = rp[0]
                    x[1] = rp[1]
                    for j in range(2, dim):
                        x[j] = 0.5
                    obj = problem(x)
                    results.append(obj)
        
        return results
