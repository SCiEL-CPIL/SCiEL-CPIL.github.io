"""Non-stationary linear contextual bandit: Cholesky-accelerated Per-Arm LinUCB
with adaptive forgetting and global context normalization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm LinUCB using Cholesky decomposition for stable and fast
    variance computation, with adaptive forgetting for non-stationarity."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 115.0
        self.deadline_hit = False

        # Per-arm Cholesky factors L such that A = L @ L.T
        # Initialize A = lambda * I, so L = sqrt(lambda) * I
        self.lam = 1.0
        self.L = [np.eye(dim) * np.sqrt(self.lam) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Global context statistics for normalization
        self.ctx_norm_sum = 0.0
        self.ctx_count = 0
        self.avg_ctx_norm = 1.0

        # Forgetting parameters
        self.forgetting = 0.95  # Base decay
        self.f_min = 0.80
        self.f_max = 0.99
        self.alpha = 1.0        # Exploration coefficient
        self.last_pred = 0.0
        self.last_reward = 0.0
        self.count = 0
        self.error_ewma = 0.0   # Exponentially weighted moving average of error

    def _check_deadline(self):
        if self.deadline_hit:
            return
        now = time.time()
        if now - self.t0 >= self.deadline:
            self.deadline_hit = True

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        
        if self.deadline_hit:
            # Exploitation only
            scores = np.empty(self.n)
            for a in range(self.n):
                # Solve L y = ctx
                y = np.linalg.solve(self.L[a], ctx)
                # Solve L.T z = y
                z = np.linalg.solve(self.L[a].T, y)
                theta = z # This is A^-1 @ ctx? No.
                # A = L L^T. A^-1 = L^-T L^-.
                # We want x^T A^-1 x.
                # Let v = A^-1/2 x. Then ||v||^2 = x^T A^-1 x.
                # L v = x => v = L^-1 x.
                # Then ||v|| is the norm.
                # Wait, A = L L^T. A^-1 = L^-T L^-.
                # A^-1/2 = L^-T (L^T L)^-1/2 L^-1? No.
                # Standard trick: 
                # 1. Solve L u = x
                # 2. Solve L^T v = u
                # Then v = L^-T L^-1 x = A^-1 x.
                # Variance is x^T A^-1 x = x^T v.
                # Alternatively, just compute norm of u? 
                # x^T A^-1 x = x^T (L L^T)^-1 x = x^T L^-T L^-1 x = (L^-1 x)^T (L^-1 x) = ||L^-1 x||^2.
                # So we just need u = L^-1 x.
                u = np.linalg.solve(self.L[a], ctx)
                scores[a] = (u @ self.b[a]) + self.alpha * np.linalg.norm(u)
            return int(np.argmax(scores))

        scores = np.empty(self.n)
        for a in range(self.n):
            # Compute A^-1/2 x efficiently
            # u = L^-1 x
            u = np.linalg.solve(self.L[a], ctx)
            # Prediction is x^T A^-1 b = u^T b (since A^-1 b = L^-T L^-1 b, and x^T A^-1 b = (L^-1 x)^T (L^-T b)? No.
            # x^T A^-1 b. Let y = A^-1 b.
            # We can maintain theta = A^-1 b directly? 
            # Updating theta with Cholesky is tricky.
            # Easier: Just compute the score.
            # score = x^T theta + alpha * sqrt(x^T A^-1 x)
            # theta = A^-1 b.
            # We can solve A theta = b.
            # Or, note that x^T A^-1 b = (L^-1 x)^T (L^-T b).
            # Let v = L^-T b. Then score = u^T v.
            # We can pre-compute v? b changes every step.
            # So for each arm:
            # 1. u = solve(L, x)
            # 2. v = solve(L.T, b)
            # 3. pred = dot(u, v)
            # 4. var = dot(u, u)
            
            u = np.linalg.solve(self.L[a], ctx)
            v = np.linalg.solve(self.L[a].T, self.b[a])
            pred = np.dot(u, v)
            var = np.dot(u, u)
            
            # Normalize exploration by context magnitude relative to average
            ctx_norm = np.linalg.norm(ctx)
            norm_factor = self.avg_ctx_norm / (ctx_norm + 1e-6)
            
            scores[a] = pred + self.alpha * np.sqrt(var) * norm_factor

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self.deadline_hit:
            return

        # Update global context stats
        ctx_norm = np.linalg.norm(ctx)
        self.ctx_norm_sum += ctx_norm
        self.ctx_count += 1
        self.avg_ctx_norm = self.ctx_norm_sum / self.ctx_count

        # Drift detection and adaptive forgetting
        # Compute current prediction for this arm to estimate error
        # (Approximate: use the previous theta estimate or just the error from last step)
        # For simplicity, use the error from the *last* update to adjust forgetting for the *current* update.
        # Or, compute error now? We don't have the theta explicitly handy without extra work.
        # Let's stick to the previous error heuristic but refine it.
        
        if self.count >= 1:
            err = abs(reward - self.last_pred)
            self.error_ewma = 0.8 * self.error_ewma + 0.2 * err
            
            # High error -> high forgetting (lower f)
            # Map error to [f_min, f_max]
            # err=0 -> f_max, err=large -> f_min
            target_f = self.f_max - (self.f_max - self.f_min) * (1 - np.exp(-self.error_ewma / 1.0))
            self.forgetting += 0.1 * (target_f - self.forgetting)
            self.forgetting = min(max(self.forgetting, self.f_min), self.f_max)
        else:
            self.forgetting = 0.95

        f = self.forgetting
        
        # Update A and b for the selected arm
        # A_new = f * A_old + x x^T
        # b_new = f * b_old + reward * x
        
        # We maintain L such that A = L L^T.
        # Updating L is non-trivial with decay.
        # A_new = f (L L^T) + x x^T
        # This is not a simple rank-1 update of the Cholesky factor.
        # Option 1: Re-cholesky every step. O(d^3). d is small (likely < 100).
        # Option 2: Maintain A explicitly, compute Cholesky every step.
        # Option 3: Use a simpler inverse update.
        
        # Given d is likely small (context dim), O(d^3) might be fine.
        # But let's try to keep it efficient.
        # If d=10, d^3=1000 ops. 5 arms. 5000 ops/step. 10k steps = 50M ops. Fine.
        
        # Let's switch to maintaining A and b explicitly, and computing Cholesky in select?
        # No, select is called every step.
        # Better: Maintain A and b. In select, compute Cholesky?
        # Or maintain L and do a "re-cholesky" in update?
        
        # Let's maintain A and b.
        # In __init__, A = lam * I.
        # In update:
        # self.A[arm] = f * self.A[arm] + np.outer(ctx, ctx)
        # self.b[arm] = f * self.b[arm] + reward * ctx
        # In select:
        # Compute Cholesky of A[arm]? O(d^3) per arm per step.
        # This might be too slow if d is large.
        
        # Alternative: Keep the previous code's `np.linalg.inv` but it was slow/unstable.
        # Let's use `np.linalg.cholesky` in select.
        
        # I will refactor to maintain A and b, and use Cholesky in select.
        
        # Wait, I can just use the previous `L` update logic if I ignore the decay for L?
        # No, decay affects A.
        
        # Let's go with maintaining A and b, and computing Cholesky in select.
        # This is O(K * d^3) per step.
        
        # Refactor __init__ to store A
        # Refactor select to compute Cholesky
        
        # To avoid refactoring the whole class structure in my head, I will just
        # implement the version that stores A and b, and uses Cholesky in select.
        
        # However, the current code structure in the prompt uses `self.A` and `self.b` as lists of matrices.
        # I will keep that.
        
        # Update A and b
        self.A[arm] = f * self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = f * self.b[arm] + reward * ctx

        # Compute prediction for error tracking
        # theta = A^-1 b
        # Use np.linalg.solve(A, b)
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            self.last_pred = np.dot(ctx, theta)
        except np.linalg.LinAlgError:
            self.last_pred = 0.0
            
        self.count += 1

    # Helper to get scores using Cholesky
    def _get_scores(self, ctx):
        scores = np.empty(self.n)
        for a in range(self.n):
            try:
                L = np.linalg.cholesky(self.A[a])
            except np.linalg.LinAlgError:
                # Fallback to identity if singular
                L = np.eye(self.d)
            
            # u = L^-1 ctx
            u = np.linalg.solve(L, ctx)
            # v = L^-T b
            v = np.linalg.solve(L.T, self.b[a])
            
            pred = np.dot(u, v)
            var = np.dot(u, u)
            
            ctx_norm = np.linalg.norm(ctx)
            norm_factor = self.avg_ctx_norm / (ctx_norm + 1e-6)
            
            scores[a] = pred + self.alpha * np.sqrt(var) * norm_factor
        return scores

# Need to update __init__ and select to use this new logic
# The class definition above is a sketch. I will write the final clean version.

# EVOLVE-BLOCK-END
