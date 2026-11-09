"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # For DTLZ1, the Pareto front is characterized by:
        # x[0] and x[1] determine the position on the front via:
        #   f0 = 0.5 * x[0] * (1 + g)
        #   f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        #   f2 = 0.5 * x[0] * x[1] * (1 + g)
        # where g depends on x[2:]. To minimize g, we need to find optimal x[2:].
        #
        # Strategy:
        # 1. Find optimal x[2:] by minimizing g (sum of objectives at a neutral point)
        # 2. Sample many points on the Pareto front by varying x[0], x[1]
        # 3. Use remaining budget for local refinement of x[2:]

        rng = np.random.default_rng(42)
        best_x2 = np.full(dim - 2, 0.5)
        best_g_val = np.inf
        
        # Phase 1: Find optimal x[2:] configuration
        # Probe candidates for x[2:]
        candidates = [
            np.zeros(dim - 2),
            np.ones(dim - 2),
            np.full(dim - 2, 0.5),
        ]
        for _ in range(8):
            candidates.append(rng.random(dim - 2))
            
        for x2 in candidates:
            if time.time() > deadline:
                break
            x_test = np.concatenate([[0.5, 0.5], x2])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_val = np.sum(f)
            if g_val < best_g_val:
                best_g_val = g_val
                best_x2 = x2.copy()
        
        # Local search to refine x[2:]
        n_local = min(15, self.budget - 12)
        for _ in range(n_local):
            if time.time() > deadline:
                break
            x2_pert = best_x2 + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
            x2_pert = np.clip(x2_pert, lo[2:], hi[2:])
            x_test = np.concatenate([[0.5, 0.5], x2_pert])
            x_test = np.clip(x_test, lo, hi)
            f_test = problem(x_test)
            g_test = np.sum(f_test)
            if g_test < best_g_val:
                best_g_val = g_test
                best_x2 = x2_pert.copy()
        
        evaluated = 12 + n_local
        
        # Phase 2: Sample points on the Pareto front
        # The front is parameterized by (alpha, beta) where alpha + beta = 1, alpha, beta >= 0
        # f0 = 0.5 * alpha * (1 + g_min)
        # f1 = 0.5 * beta * (1 - alpha) * (1 + g_min)  -- wait, let me re-derive
        # Actually for DTLZ1 with m=3:
        # f0 = 0.5 * x0 * (1 + g)
        # f1 = 0.5 * x0 * (1 - x1) * (1 + g)  
        # f2 = 0.5 * x0 * x1 * (1 + g)
        # On the Pareto front, g = g_min, and we vary x0, x1 with x0 + x1 = 1? No...
        # Actually the front is: f_i = 0.5 * x0 * prod_{j=1}^{i-1}(1-x_j) * prod_{j=i}^{m-2} x_j * (1+g)
        # For m=3: f0=0.5*x0*(1+g), f1=0.5*x0*(1-x1)*(1+g), f2=0.5*x0*x1*(1+g)
        # Sum: f0+f1+f2 = 0.5*x0*(1+g) = const on front
        # So we need x0 = 2/(1+g_min) to get the standard front
        # Then f0 = 1/(1+g_min), f1 = (1-x1)/(1+g_min), f2 = x1/(1+g_min)
        # But we want f0+f1+f2 = 1, so 1/(1+g_min) + (1-x1)/(1+g_min) + x1/(1+g_min) = (2)/(1+g_min)
        # Hmm, let me just sample x0, x1 directly and evaluate
        
        recorded_F = []
        
        # First, evaluate the ideal point (x0=0, x1=0)
        x_ideal = np.concatenate([[0.0, 0.0], best_x2])
        x_ideal = np.clip(x_ideal, lo, hi)
        f_ideal = problem(x_ideal)
        recorded_F.append(f_ideal)
        evaluated += 1
        
        # Sample a grid of (x0, x1) values on the front
        # The Pareto front for DTLZ1 (m=3) is a 2D simplex
        # Points satisfy: f0 + f1 + f2 = 0.5 * x0 * (1 + g_min)
        # We want to cover the entire front
        
        # Use remaining budget for front sampling
        remaining = self.budget - evaluated
        if remaining > 0:
            # Create a stratified sample of (x0, x1) pairs
            # x0 controls the "radius" and x1 controls the "angle" on the 2D front
            # For the standard DTLZ1 front, x0 should be around 1.0 (scaled)
            
            # Let's sample x0 in [0, 1] and x1 in [0, 1] with x0 + x1 <= 1
            # Actually, the front is: for any (a, b) with a >= 0, b >= 0, a + b <= 1:
            #   f0 = a * (1 + g_min)
            #   f1 = b * (1 - a) * (1 + g_min)  -- this doesn't seem right
            
            # Let me just use the actual problem function to evaluate points
            # Parameterize: x0 = r * cos(theta), x1 = r * sin(theta) for r in [0,1], theta in [0, pi/2]
            # But actually for DTLZ1, the front is flat, so we just need to vary x0, x1
            
            # Sample n points uniformly on the 2D simplex {x0 >= 0, x1 >= 0, x0 + x1 <= 1}
            n_samples = min(remaining - 5, 150)  # leave some budget for refinement
            for i in range(n_samples):
                if time.time() > deadline:
                    break
                # Uniform sampling on simplex: use Dirichlet(1,1,1)
                u1 = rng.random()
                u2 = rng.random()
                if u1 + u2 > 1:
                    u1, u2 = 1 - u1, 1 - u2
                x0 = u1
                x1 = u2
                x_sample = np.concatenate([[x0, x1], best_x2])
                x_sample = np.clip(x_sample, lo, hi)
                f_sample = problem(x_sample)
                recorded_F.append(f_sample)
                evaluated += 1
            
            # Add a few more targeted samples at corners and edges
            corner_points = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.33, 0.33],
                [0.66, 0.33],
                [0.33, 0.66],
            ]
            for cp in corner_points:
                if time.time() > deadline:
                    break
                if evaluated >= self.budget:
                    break
                x_cp = np.concatenate([cp, best_x2])
                x_cp = np.clip(x_cp, lo, hi)
                f_cp = problem(x_cp)
                # Check if this point is already recorded (avoid duplicates)
                is_dup = False
                for existing in recorded_F:
                    if np.allclose(existing, f_cp, atol=1e-10):
                        is_dup = True
                        break
                if not is_dup:
                    recorded_F.append(f_cp)
                    evaluated += 1
        
        return recorded_F
# EVOLVE-BLOCK-END
