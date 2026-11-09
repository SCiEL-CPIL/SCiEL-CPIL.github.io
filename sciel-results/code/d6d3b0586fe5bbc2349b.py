"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) approach:
  - CMA-ES is a state-of-the-art black-box optimizer for smooth, non-convex
    continuous functions.
  - It adapts a multivariate normal distribution to the shape of the landscape,
    handling ill-conditioning and rotations effectively.
  - This is a structural change from the previous Nelder-Mead (simplex) approach,
    moving from a local direct search method to a population-based evolutionary
    strategy with covariance adaptation.
  - We use a compact, efficient implementation suitable for small dimensions
    (dim=5) and tight budgets (500 evaluations).

Strategy:
  1. Initialize mean at center of box, covariance as identity, sigma appropriately.
  2. Iteratively sample population, evaluate, rank, update mean and covariance.
  3. Track global best.
  4. Stop when budget or time limit is reached.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14 with hidden affine transform."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, budget, dim, deadline, start_time):
        """
        Run CMA-ES optimization.
        Returns (best_x, best_f).
        """
        # Parameters
        # Population size lambda
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights for recombination
        mu = lambda_ // 2
        # Weights: linear weights normalized
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial mean and sigma
        # Start at origin, but maybe slightly randomized to avoid symmetry issues?
        # For BBOB, origin is often a good start if shift is small, but here shift is hidden.
        # However, CMA-ES will move. Let's start at 0.
        mean = np.zeros(dim)
        # Initial sigma: heuristic for box size [-5, 5]
        # A standard deviation of 1.0 covers most of the box in 1D.
        # For dim=5, starting sigma=1.0 is reasonable.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky factor (or just use sqrtm for simplicity in small dim, but cholesky is better)
        # We will update C and use np.linalg.cholesky each step for sampling
        # Or maintain B and D. Let's stick to C for clarity, dim=5 is small.
        
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        
        # Best tracking
        best_x = mean.copy()
        best_f = float(func(best_x))
        calls_counter = 1
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Main loop
        while calls_counter < budget and time.time() < deadline:
            # 1. Sample population
            # Get Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C becomes singular
                B = np.eye(dim)
            
            # Sample lambda_ points
            z = np.random.randn(lambda_, dim)
            # Transform to x
            pop = mean[:, np.newaxis] + sigma * (z @ B.T)
            
            # Clip to box (optional, but keeps values finite)
            # BBOB functions usually handle clipping, but let's ensure we stay in bounds
            # Actually, the problem says "func(x) clips...". So we can pass unclipped, 
            # but it's safer to clip for numerical stability if the function is weird.
            # The prompt says "func(x) clips a finite length-5 vector to the box".
            # So we can just pass pop.
            
            # 2. Evaluate
            fvals = np.array([float(func(x)) for x in pop])
            calls_counter += lambda_
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
            
            # 3. Rank
            order = np.argsort(fvals)
            y = pop[order]
            fvals_sorted = fvals[order]
            
            # 4. Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * y[:mu], axis=0)
            
            # 5. Update step size control path ps
            # hsig condition
            hsig = np.sqrt((1 - np.exp(-2 / cs)) / np.exp(-2 / cs) - 1)
            # norm of ps
            # We need B^-1 to transform mean difference to normal space
            # B is cholesky of C, so C = B @ B.T. 
            # The transformation is C^{-1/2} = B^{-T} @ B^{-1}
            # Let's compute inv_B
            try:
                inv_B = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                inv_B = np.eye(dim)
            
            # Transform mean difference
            diff = (mean - mean_old) / sigma
            # C^{-1/2} diff = inv_B.T @ (inv_B @ diff) ? No.
            # If C = B B^T, then C^{-1/2} is not simply related to B unless we use specific decomposition.
            # Standard CMA-ES uses B and D from C = B diag(D^2) B^T.
            # For simplicity in this compact version, let's use the approximation 
            # or just check convergence via sigma and f-range.
            # Actually, for dim=5, we can afford to compute eigen-decomposition or just use
            # a simpler sigma update.
            
            # Simplified Sigma Update (less rigorous but often effective for small dims)
            # If we don't implement full hsig/ps/pc rigorously, we can use a simpler
            # adaptation rule.
            # Let's try a standard but slightly simplified CMA-ES update for C.
            
            # Update C
            # Delta C = c1 * (mean_old - mean) @ (mean_old - mean).T + cmu * sum(weights * (y - mean_old) @ (y - mean_old).T)
            # Note: y is sorted, mean_old is the old mean.
            # The formula is usually based on the ranked differences from the OLD mean.
            
            # Calculate differences from old mean
            # y are the new points.
            # diff_y = y[:mu] - mean_old
            
            # For robustness, let's use the standard update:
            # C = (1 - c1 - cmu) * C + c1 * (mean - mean_old) @ (mean - mean_old).T 
            #     + cmu * sum(weights * (y_i - mean_old) @ (y_i - mean_old).T)
            
            # Wait, the standard update uses the "rank-one" and "rank-mu" updates.
            # Rank-one: (mean - mean_old)
            # Rank-mu: sum w_i (y_i - mean_old)(y_i - mean_old)^T
            
            # Let's compute these.
            diff_mean = mean - mean_old
            rank_one = np.outer(diff_mean, diff_mean)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_y = y[i] - mean_old
                rank_mu += weights[i] * np.outer(diff_y, diff_y)
            
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            C += 1e-16 * np.eye(dim)
            
            # Update Sigma
            # Simple rule: if fvals improved, keep sigma; else reduce?
            # Better: use the standard ps/pc logic.
            # Let's implement a simplified version of the sigma adaptation.
            # If the average distance of the selected points from the old mean is smaller
            # than expected, decrease sigma.
            
            # Expected norm of a standard normal in dim D is chi_n.
            # The norm of the step in the transformed space is sigma * ||C^{-1/2} (mean - mean_old)||
            # This is complex. Let's use a heuristic:
            # If the best function value in the population is significantly better than the old best,
            # we might want to increase sigma slightly or keep it.
            # If we are stagnating, decrease sigma.
            
            # A common simple adaptation:
            # sigma *= exp(cs * (||ps|| / chi_n - 1))
            # We need ps.
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (mean - mean_old)/sigma
            
            # We need C^{-1/2}. Let's compute eigenvalues/vectors of C.
            # For dim=5, this is cheap.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
            
            # Update ps
            # diff_mean is mean - mean_old
            # step = diff_mean / sigma
            # transformed_step = C_inv_sqrt @ step
            step = diff_mean / sigma
            transformed_step = C_inv_sqrt @ step
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * transformed_step
            
            # Update pc
            # pc = (1 - cc) * pc + cc * hsig * (mueff * C_inv_sqrt @ step)
            # hsig is the Heaviside step function based on ||ps||/chi_n
            hsig = 1.0 if np.linalg.norm(ps) / chi_n < (1 + 1.0/(2*cs)) * np.sqrt((dim + 1) / (dim - 1)) else 0.0
            # Actually, standard hsig is:
            # hsig = 1 if ||ps||/chi_n < (1 + 1/(2*cs)) * (1 + 1/(dim-1)) ?
            # Let's use the standard formula:
            # hsig = 1 if ||ps||/chi_n < (1 + 1/(2*cs)) * np.sqrt((dim+1)/(dim-1))
            # Wait, the factor is (1 + 1/(2*cs)) * (1 + 1/(dim-1))? No.
            # The condition is ||ps||/chi_n < (1 + 1/(2*cs)) * (1 + 1/(dim-1)) is for something else.
            # Standard: hsig = 1 if ||ps||/chi_n < (1 + 1/(2*cs)) * np.sqrt((dim+1)/(dim-1))
            # Let's assume hsig=1 for now to simplify, or use a robust check.
            # If we skip pc update, C adaptation is slower but still works.
            # Let's include pc update for better performance.
            
            # Recalculate hsig properly
            hsig_val = 1.0 if np.linalg.norm(ps) / chi_n < (1 + 1.0 / (2.0 * cs)) * np.sqrt((dim + 1.0) / (dim - 1.0)) else 0.0
            
            pc = (1 - cc) * pc + hsig_val * cc * mueff * (C_inv_sqrt @ step)
            
            # Update C with pc
            # The rank-one update for C should include the pc term?
            # Standard CMA-ES update for C:
            # C = (1 - c1 - cmu - c1*cc*(2-cs)*mueff/2) * C + ...
            # This is getting complicated. 
            # Given dim=5 and budget=500, a slightly less rigorous CMA-ES 
            # (omitting pc or simplifying) might be sufficient and faster.
            
            # Let's stick to the Rank-One + Rank-Mu update for C as implemented above, 
            # and use the ps-based sigma update.
            
            # Sigma update
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Convergence check
            # If sigma is very small or f-range is small, stop
            if sigma < 1e-12 or (fvals_sorted[-1] - fvals_sorted[0] < 1e-12):
                break

        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # 10s margin before 330s kill
        dim = self.dim
        budget = self.budget
        
        # Run CMA-ES
        best_x, best_f = self._cma_es(func, budget, dim, deadline, start)
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        return best_x, float(best_f)
