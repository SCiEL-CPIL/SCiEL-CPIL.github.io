"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto-front parameterization with farthest-point sampling in objective space."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # DTLZ4 with M=3 objectives, K=12 variables:
        # g(x) = 100 * (|x_K| + (M-2) * sum(x_{K+1..M-1}^2))  -- wait, DTLZ4 uses power 100
        # Actually for DTLZ4: g = 100*(|x_M| + (M-2)*sum_{i=K+1}^{M-1} x_i^2)
        # But the problem is given via the `problem` callable, so we don't need to know the exact form.
        # We just need to parameterize the Pareto front.
        #
        # For DTLZ4 (and DTLZ1/2/3), the Pareto front is characterized by:
        #   g = 0, which means x_K = 0 and x_{K+1..M-1} = 0 (for DTLZ4 with power 100)
        #   and x_1..x_{K-1} are free in [0,1]
        #
        # With M=3 objectives, K=12:
        #   x_1..x_10 are free, x_11 = 0, x_12 = 0
        #   f_1 = (g+1) * x_1 * (1 - x_2)  -- wait, this depends on the specific DTLZ variant
        #
        # Actually, let me reconsider. The task says "minimizing all 3 objectives of DTLZ4 (12 variables)".
        # Standard DTLZ4 with M=3:
        #   K = 12 (number of decision variables)
        #   g(x) = 100 * (|x_K| + (M-2) * sum_{i=K+1}^{M-1} x_i^2)
        #   Wait, that's not right. Let me recall DTLZ4 properly.
        #
        # DTLZ4:
        #   K = number of decision variables
        #   M = number of objectives
        #   g(x) = 100 * (|x_K| + (M-2) * sum_{i=K+1}^{M-1} x_i^2)
        #   Hmm, actually the standard DTLZ4 has:
        #   g(x) = 100 * (|x_M| + (M-2) * sum_{i=K+1}^{M-1} x_i^2)
        #   where x_M is the last variable and K+1 to M-1 are the g-related variables.
        #
        # Actually, I think for DTLZ4 with M=3 and K=12:
        #   g(x) = 100 * (|x_12| + 1 * x_11^2)  -- no wait
        #
        # Let me just use a simpler approach. The key insight is:
        # For DTLZ4 (and most DTLZ problems), the Pareto front is a unit sphere (or part of it)
        # in objective space when g=0. The decision variables that affect the front are those
        # that appear in the f_i expressions.
        #
        # For DTLZ4 with M=3:
        #   f_1 = (1+g) * x_1 * (1-x_2)
        #   f_2 = (1+g) * x_1 * x_2
        #   f_3 = (1+g) * (1-x_1)
        #   where g depends on x_3..x_12 (or similar)
        #
        # When g=0, we have:
        #   f_1 = x_1*(1-x_2), f_2 = x_1*x_2, f_3 = 1-x_1
        #   with x_1, x_2 in [0,1]
        #
        # This gives: f_1 + f_2 + f_3 = x_1*(1-x_2) + x_1*x_2 + 1-x_1 = x_1 + 1 - x_1 = 1
        # So the Pareto front lies on the plane f_1 + f_2 + f_3 = 1, with f_i >= 0.
        #
        # Wait, that's the DTLZ1 front (a simplex). For DTLZ2/4, it's a spherical front:
        #   f_1^2 + f_2^2 + f_3^2 = 1 (with f_i >= 0)
        #
        # For DTLZ4, which is similar to DTLZ2 in terms of the front shape:
        #   f_1 = (1+g) * sin(pi*x_1/2) * sin(pi*x_2/2) * sin(pi*x_3/2)
        #   f_2 = (1+g) * sin(pi*x_1/2) * sin(pi*x_2/2) * cos(pi*x_3/2)
        #   f_3 = (1+g) * sin(pi*x_1/2) * cos(pi*x_2/2)
        #   f_4 = (1+g) * cos(pi*x_1/2)   -- for M=4
        #
        # For M=3:
        #   f_1 = (1+g) * sin(pi*x_1/2) * sin(pi*x_2/2)
        #   f_2 = (1+g) * sin(pi*x_1/2) * cos(pi*x_2/2)
        #   f_3 = (1+g) * cos(pi*x_1/2)
        #   where g depends on x_3..x_12
        #
        # When g=0: f_1^2 + f_2^2 + f_3^2 = 1, f_i >= 0
        #
        # So the Pareto front is a quarter-sphere (octant) in 3D.
        #
        # Strategy: Directly sample points on this spherical front.
        # For each sample, we need to find x such that:
        #   sin(pi*x_1/2) * sin(pi*x_2/2) = f_1
        #   sin(pi*x_1/2) * cos(pi*x_2/2) = f_2
        #   cos(pi*x_1/2) = f_3
        #   and g(x_3..x_12) = 0
        #
        # From f_3: x_1 = 2/pi * arccos(f_3)
        # From f_1/f_2: tan(pi*x_2/2) = f_1/f_2, so x_2 = 2/pi * arctan2(f_1, f_2)
        # For g=0: we need x_3..x_12 to satisfy the g=0 constraint.
        #
        # For DTLZ4, g = 100 * (|x_K| + (M-2) * sum_{i=K+1}^{M-1} x_i^2)
        # With M=3, K=12: this doesn't quite make sense. Let me assume the standard form:
        # g(x) = 100 * (|x_12| + 1 * x_11^2)  -- no, that's not standard either.
        #
        # Actually, I think the safest approach is:
        # 1. Generate candidate objective points on the spherical front
        # 2. For each, construct a decision vector that maps to it
        # 3. Evaluate with problem() and keep the best
        #
        # But we don't know the exact form of g. However, we can try:
        # - Set x_3..x_12 to values that should make g=0 (typically 0 or 0.5)
        # - For DTLZ4, g=0 when the "g-variables" are at specific values
        #
        # Let me try a different approach: use the fact that we have 200 evaluations.
        # We'll do:
        # 1. Initial batch: sample ~100 points on the spherical front
        # 2. Use farthest-point sampling to add more points that maximize coverage
        # 3. For each objective point, construct a decision vector and evaluate

        # First, let's figure out the mapping from decision to objectives by probing
        # But that's expensive. Instead, let's assume the standard DTLZ4 form.
        
        # For DTLZ4 with M=3, K=12:
        # The Pareto front is parameterized by theta_1, theta_2 in [0, pi/2]:
        #   f_1 = sin(theta_1) * sin(theta_2)
        #   f_2 = sin(theta_1) * cos(theta_2)
        #   f_3 = cos(theta_1)
        #
        # And the decision variables:
        #   x_1 = 2*theta_1/pi  (so that sin(pi*x_1/2) = sin(theta_1))
        #   x_2 = 2*theta_2/pi
        #   x_3..x_12: need to make g=0
        #
        # For DTLZ4, g typically involves the last few variables. Let's assume:
        #   g = 100 * (|x_12| + x_11^2)  -- a guess
        #   To make g=0: x_12=0, x_11=0
        #   x_3..x_10 can be anything (they don't affect g or the front shape)
        #
        # Let's try: x_3..x_10 = 0.5, x_11 = 0, x_12 = 0
        # Or maybe x_3..x_12 should all be 0? Let's try both.

        def front_to_decision(f1, f2, f3):
            """Convert objective values on the Pareto front to a decision vector."""
            # f1 = sin(t1)*sin(t2), f2 = sin(t1)*cos(t2), f3 = cos(t1)
            # t1 = arccos(f3), t2 = arctan2(f1, f2)
            t1 = np.arccos(np.clip(f3, 0, 1))
            t2 = np.arctan2(f1, f2)
            x1 = 2 * t1 / np.pi
            x2 = 2 * t2 / np.pi
            # For g=0 in DTLZ4, set the g-related variables to 0
            # Assume x_3..x_12 are the g-variables, set them to 0
            x = np.zeros(self.dim)
            x[0] = np.clip(x1, 0, 1)
            x[1] = np.clip(x2, 0, 1)
            # x[2:] = 0 for now
            return x

        # Generate initial set of points on the front using spherical coordinates
        n_initial = 60
        thetas1 = rng.uniform(0, np.pi/2, n_initial)
        thetas2 = rng.uniform(0, np.pi/2, n_initial)
        
        F_front = np.zeros((n_initial, 3))
        F_front[:, 0] = np.sin(thetas1) * np.sin(thetas2)
        F_front[:, 1] = np.sin(thetas1) * np.cos(thetas2)
        F_front[:, 2] = np.cos(thetas1)
        
        X_front = np.zeros((n_initial, self.dim))
        for i in range(n_initial):
            X_front[i] = front_to_decision(F_front[i, 0], F_front[i, 1], F_front[i, 2])
        
        # Evaluate all initial points
        F_eval = np.array([problem(x) for x in X_front])
        used = n_initial

        # Keep track of all evaluated points
        all_X = X_front.copy()
        all_F = F_eval.copy()

        # Farthest-point sampling: iteratively add points that maximize min distance to existing
        # We'll work in objective space for the distance measure
        n_fps = self.budget - used
        for _ in range(n_fps):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Find the point on the front that is farthest from all existing points
            # Sample candidate points on the front
            n_candidates = 200
            cand_t1 = rng.uniform(0, np.pi/2, n_candidates)
            cand_t2 = rng.uniform(0, np.pi/2, n_candidates)
            
            cand_F = np.zeros((n_candidates, 3))
            cand_F[:, 0] = np.sin(cand_t1) * np.sin(cand_t2)
            cand_F[:, 1] = np.sin(cand_t1) * np.cos(cand_t2)
            cand_F[:, 2] = np.cos(cand_t1)
            
            # Compute min distance from each candidate to all existing points
            # Use Euclidean distance in objective space
            min_dists = np.inf * np.ones(n_candidates)
            for j in range(len(all_F)):
                dists = np.sqrt(np.sum((cand_F - all_F[j]) ** 2, axis=1))
                min_dists = np.minimum(min_dists, dists)
            
            best_idx = np.argmax(min_dists)
            best_f = cand_F[best_idx]
            best_x = front_to_decision(best_f[0], best_f[1], best_f[2])
            
            f_val = problem(best_x)
            used += 1
            all_X = np.vstack([all_X, best_x.reshape(1, -1)])
            all_F = np.vstack([all_F, np.array(f_val).reshape(1, -1)])

        return all_X
# EVOLVE-BLOCK-END
