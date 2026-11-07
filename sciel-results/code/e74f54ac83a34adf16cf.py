import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES for BBOB f11 (Discus).
    Uses a deterministic grid initialization to find the basin,
    then refines with CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                raise StopIteration
            calls += 1
            # Clip to box as per contract, though func does it too
            x_clipped = np.clip(x, -5, 5)
            try:
                f = func(x_clipped)
            except Exception:
                # If func fails, return a large value to discourage this point
                # But the contract says "an exception later keeps the calls made"
                # It's better to let the exception propagate if it's a real error?
                # "an exception before the first call, makes the whole result invalid"
                # "an exception later keeps the calls made"
                # So if an exception happens, we should probably stop and return best so far?
                # But the evaluator says "exception later keeps the calls made".
                # This implies the program should terminate or continue?
                # "More than 500 calls, or an exception before the first call, makes the whole result invalid"
                # It doesn't explicitly say what happens if an exception happens later.
                # "an exception later keeps the calls made" suggests the result might still be scored?
                # But usually, exceptions crash the script.
                # To be safe, I will catch exceptions in wrapped_func and return a large value,
                # effectively treating the point as invalid, but continuing the search?
                # No, if func raises, it's likely a bug in my code or the evaluator.
                # I will assume func is robust.
                raise
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # 1. Deterministic Grid Initialization
        # Minimizer lies in [-4, 4]^5
        # Use a 2x2x2x2x2 grid (32 points) to ensure we hit the basin
        grid_points = np.array(np.meshgrid(*([-4.0, 4.0] * self.dim))).T.reshape(-1, self.dim)
        
        for p in grid_points:
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
            try:
                wrapped_func(p)
            except StopIteration:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen with budget 500)
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        # 2. CMA-ES Refinement
        remaining_budget = self.budget - calls
        
        if remaining_budget > 10:
            self._run_cmaes(wrapped_func, best_x, remaining_budget, deadline)
        
        # Ensure we return the best found
        return best_x, best_f

    def _run_cmaes(self, func, x0, budget, deadline):
        n = self.dim
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        
        # If budget is too small for even one generation, do simple random walk
        if budget < lambda_pop:
            for _ in range(budget):
                if time.time() > deadline:
                    break
                y = x0 + 0.5 * np.random.randn(n)
                try:
                    func(y)
                except StopIteration:
                    break
            return

        m = x0.copy()
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        
        # CMA-ES parameters
        mu = lambda_pop // 2
        weights = np.zeros(lambda_pop)
        for i in range(mu):
            weights[i] = np.log((mu + 1) / 2) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + n / 3) / (n + 4 + 2 / n)
        cs = (n + 2) / (n + 6)
        cmu = 2 / ((n + 1.3) ** 2 + n)
        dsigma = 1 + 2 * max(0, (n + 1) / 3 - n) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        calls_used = 0
        best_x_local = m.copy()
        best_f_local = float('inf')
        
        # We need to track the global best from the wrapper, but wrapper updates non-locals.
        # The wrapper updates self's non-locals. So we just need to run the loop.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate samples
            B = np.linalg.cholesky(C).T
            ys = np.random.randn(lambda_pop, n)
            xs = m + (sigma * (ys @ B.T)).T
            fs = []
            
            for x in xs:
                if calls_used >= budget or time.time() > deadline:
                    break
                try:
                    f = func(x)
                    calls_used += 1
                    fs.append(f)
                except StopIteration:
                    break
            
            if len(fs) < lambda_pop:
                # Incomplete generation, stop
                break
            
            # Sort
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            ys_sorted = ys[order]
            
            # Update mean
            m_new = np.sum(weights[:mu, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update pc
            y_sigma = (m_new - m) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(pc) / np.sqrt(mu_eff) - chi_n))
            
            # Update C
            rank_mu = np.outer(y_sigma, y_sigma)
            rank_one = np.zeros((n, n))
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) < (1.4 + cc) * chi_n:
                rank_one = np.outer(pc, pc)
            
            C = (1 - mu_eff * cmu) * C + cmu * rank_mu + (1 - mu_eff * cmu) * cs * (2 - cs) * rank_one
            
            # Check positive definiteness
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Reset if singular
                C = np.eye(n)
                sigma = 1.0
                pc = np.zeros(n)
            
            m = m_new
            
            if sigma < 1e-6:
                break

        return
