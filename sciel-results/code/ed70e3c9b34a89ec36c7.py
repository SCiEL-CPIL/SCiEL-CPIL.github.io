import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit

        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = int(4 + 3 * np.log(dim))
        # For dim=5, lam = 4 + 3*1.6 = 8.8 -> 8
        
        # Initial sigma (step size)
        # Domain is [-5, 5], minimizer in [-4, 4].
        # Start with a reasonably large step to explore, then adapt.
        sigma = 1.0
        
        # Center of the distribution
        # Uniform random start in the center of the box
        x_center = rng.uniform(-3.0, 3.0, dim)
        
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters (standard CMA-ES values)
        mu = lam // 2  # Number of best solutions used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping for step size control
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Hessian eigenvalue check threshold (approximate)
        hsig = (1.4 + 2 / (dim + 1)) * (1 - np.exp(-1 / c_sigma))
        
        best_x = x_center.copy()
        best_f = np.inf
        eval_count = 0
        
        # Initial evaluation of the center
        f_center = func(x_center)
        eval_count += 1
        best_f = f_center
        best_x = x_center.copy()
        
        # Main CMA-ES Loop
        while eval_count < budget and not time_check():
            # Generate population
            # Sample from multivariate normal: x = x_center + sigma * B * z
            # Where B is the matrix such that B @ B.T = C
            # We maintain B via Cholesky decomposition or eigendecomposition.
            # For simplicity and stability in small dims, we can use eigendecomposition of C.
            # However, Cholesky is faster. Let's use Cholesky if C is positive definite.
            # If C becomes singular, we restart or regularize.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-8 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    break # If it's really broken, stop

            z_samples = rng.standard_normal((lam, dim))
            # x_i = x_center + sigma * (L @ z_i)
            # Note: C = L @ L.T, so sqrt(C) is L (in a sense).
            # Actually, if C = L L^T, then x = x_center + sigma * L @ z is correct.
            pop = x_center[None, :] + sigma * (z_samples @ L.T)
            
            # Evaluate population
            f_values = np.zeros(lam)
            for i in range(lam):
                if eval_count >= budget or time_check():
                    break
                f_values[i] = func(pop[i])
                eval_count += 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = pop[i].copy()
            
            if eval_count >= budget or time_check():
                break
            
            # Sort population by fitness
            order = np.argsort(f_values)
            pop = pop[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = pop[0].copy()
                
            # Weighted mean of the mu best solutions
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update evolution path for covariance matrix adaptation
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * C^{-1/2} * (x_mean - x_center) / sigma
            # C^{-1/2} can be approximated by L^{-1} if we use the Cholesky factor from the previous step?
            # Actually, the standard update uses C^{-1/2} from the previous iteration.
            # Let's compute C_inv_sqrt using eigendecomposition for stability in this small case, 
            # or just use the inverse of L if we assume L approximates sqrt(C).
            # A common simplification is to use L^{-1} where L is the Cholesky of C.
            # Let's compute the inverse of L.
            try:
                L_inv = np.linalg.inv(L)
            except:
                break
                
            # The term is C^{-1/2} (x_mean - x_center) / sigma
            # If C = L L^T, then C^{-1/2} is not exactly L^{-1}. 
            # However, in many implementations, the update is done in the transformed space.
            # Let's stick to the standard formula:
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) / sigma * L^{-1} (x_mean - x_center)
            # Wait, the standard CMA-ES update for p_c is:
            # p_c <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * C^{-1/2} (x_mean - x_center) / sigma
            # And for p_sigma:
            # p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (x_mean - x_center) / sigma
            
            # Let's use the fact that if we sample x = x_center + sigma L z, then z = L^{-1} (x - x_center) / sigma.
            # So the "z-mean" is L^{-1} (x_mean - x_center) / sigma.
            z_mean = L_inv @ (x_mean - x_center) / sigma
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            
            # Check for stagnation / step size adaptation
            # hsig is the threshold for the norm of p_sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count // lam))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)) - 1:
                # This condition is an approximation for the hsig check.
                # Standard check: ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2k}) < (1.4 + 2/(n+1)) * sqrt((n+1)/(n-1)) - 1
                pass # We'll simplify the check below
                
            # Step size adaptation
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(n) * (1 - 1/(4n) + ...)
            # For n=5, approx 2.2
            e_norm = np.sqrt(dim) * (1 - 1.0/(4*dim)) 
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - hsig) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - x_center_old) / sigma
            
            # Calculate hsig (heuristic for rank-1 update)
            # hsig = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2k}) < (1.4 + 2/(n+1)) * sqrt((n+1)/(n-1)) - 1
            # k is the number of generations.
            # Let's approximate k as 1 for the current step in the formula, or track it.
            # Actually, the denominator is the expected norm of p_sigma if no change occurred.
            # Let's use a simpler check: if ||p_sigma|| is small, hsig=1.
            hsig = 1.0 if np.linalg.norm(p_sigma) < 1.0 else 0.0
            
            # Update C
            # First part: decay
            C = (1 - c_1 - c_mu) * C
            
            # Second part: rank-1 update from p_c
            # Note: The term is c_1 * (p_c p_c^T + (1 - hsig) * c_c * (2 - c_c) * C)
            # But p_c is in the z-space? No, p_c is updated in the z-space (standardized).
            # Wait, the standard CMA-ES update for C uses the actual steps y_i = (x_i - x_mean_old)/sigma.
            # My p_c update above used z_mean.
            # Let's re-evaluate the update formulas to be consistent.
            
            # Standard CMA-ES:
            # 1. Generate x_i = x_mean + sigma * B z_i
            # 2. Evaluate
            # 3. Sort
            # 4. x_new_mean = sum w_i x_i
            # 5. y_i = (x_i - x_mean) / sigma  (using OLD x_mean)
            # 6. p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) C^{-1/2} y_mean
            #    where y_mean = sum w_i y_i
            # 7. p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) C^{-1/2} y_mean
            # 8. sigma = sigma * exp(c_sigma/d_sigma (||p_sigma||/E||N|| - 1))
            # 9. C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - hsig) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            #    Here p_c is the path in the C^{-1/2} transformed space?
            #    Actually, in the standard formulation, p_c is in the original space?
            #    No, p_c is updated with C^{-1/2} y_mean. So p_c is in the "standardized" space?
            #    Let's check the dimensions.
            #    y_i is in original space / sigma.
            #    C^{-1/2} y_mean is in standardized space (if C is the covariance of the steps).
            #    Then p_c is in standardized space.
            #    But the update for C adds p_c p_c^T. If p_c is standardized, this term is in the standardized basis.
            #    This implies C is being updated in a mixed way?
            #    
            #    Actually, the standard implementation (Hansen) uses:
            #    p_c is a vector in R^n.
            #    The update is:
            #    p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) C^{-1/2} (x_new_mean - x_mean) / sigma
            #    C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - hsig) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
            #    
            #    This works because C is the covariance matrix of the steps in the original space.
            #    The term p_c p_c^T is added to C. Since p_c is in the "C^{-1/2}" transformed space?
            #    No, if p_c is in the transformed space, p_c p_c^T is in the transformed space.
            #    Adding it to C (original space) is incorrect unless we transform back.
            #    
            #    Let's look at a reliable implementation snippet logic.
            #    Usually, one keeps the covariance matrix C and the matrix B (sqrt(C)).
            #    The update for C is done in the original space.
            #    The vectors y_i are in the original space (scaled by 1/sigma).
            #    The term C^{-1/2} y_mean transforms y_mean into the eigenbasis of C.
            #    So p_c is in the eigenbasis.
            #    Then p_c p_c^T is in the eigenbasis.
            #    To add it to C, we need to transform it back?
            #    No, the formula is C_{new} = ... + c_1 (p_c p_c^T ...).
            #    This implies that the "C" in the formula is actually the covariance in the standardized space?
            #    No, C is the covariance in the original space.
            #    
            #    Let's simplify: Instead of implementing the full CMA-ES update from scratch which is error-prone,
            #    I will use a simplified "Adaptive" strategy that mimics CMA-ES behavior:
            #    1. Population based.
            #    2. Mean update.
            #    3. Step size adaptation based on progress.
            #    4. Covariance adaptation using the outer product of steps (rank-mu update) without the complex path updates.
            #    This is often called "Rank-Mu CMA" or similar, and is robust.
            
            # Let's restart the loop logic with a simpler, robust Rank-Mu update.
            # We already have pop and f_values sorted.
            
            # 1. Update Mean
            x_mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # 2. Calculate steps y_i (relative to old mean, scaled by sigma)
            # Note: We must use the OLD x_center for the step calculation to be consistent with the sampling.
            y_list = []
            for i in range(mu):
                y_i = (pop[i] - x_center) / sigma
                y_list.append(y_i)
            
            # 3. Update Covariance (Rank-Mu update)
            # C = (1 - c_1 - c_mu) C + c_mu * sum w_i y_i y_i^T
            # We skip the rank-1 p_c update for simplicity/robustness in this short budget.
            # This is essentially a "Covariance Matrix Adaptation" without the path memory,
            # which is still effective for dim=5.
            
            # Calculate the sum of outer products
            sum_outer = np.zeros((dim, dim))
            for i in range(mu):
                sum_outer += weights[i] * np.outer(y_list[i], y_list[i])
                
            C = (1 - c_1 - c_mu) * C + c_mu * sum_outer
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            C = C + 1e-10 * np.eye(dim)
            
            # 4. Adapt Step Size
            # Simple heuristic: If the best fitness improved significantly, keep sigma.
            # If not, reduce sigma.
            # Or use the standard CMA-ES sigma adaptation with p_sigma.
            # Let's implement a simple p_sigma adaptation.
            # y_mean = sum w_i y_i
            y_mean = np.sum(weights[:, None] * np.array(y_list), axis=0)
            
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * y_mean
            # Note: This simplified version doesn't use C^{-1/2}, so it's less optimal but stable.
            # Actually, to be consistent with the "standardized" idea, we should transform y_mean.
            # But without C^{-1/2}, we just use y_mean.
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            
            e_norm = np.sqrt(dim) * (1 - 1.0/(4*dim))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # 5. Update Center
            x_center = x_mean_new
            
            # Check for convergence/stagnation
            # If the spread of the population is very small, restart?
            # For now, rely on sigma adaptation.
            
            # If best_f is very close to 0 (or we hit a plateau), we might need to explore more.
            # The sigma adaptation should handle this.
            
            # Safety: if sigma is extremely small, we might be stuck.
            # If eval_count is high and no improvement, we could restart with a random point.
            # But CMA-ES usually handles this via sigma increase if p_sigma norm is high (which happens if we jump).
            
        return best_x, best_f
