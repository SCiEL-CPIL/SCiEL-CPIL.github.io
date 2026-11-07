import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 300s
        deadline = start_time + 300.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            return fx

        # CMA-ES Implementation
        def cma_es(start_x, sigma0, budget, rng):
            nonlocal calls_made, best_f, best_x
            
            n = dim
            # Population size
            lam = 10 
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Evolution path parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
            
            # Covariance matrix adaptation
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Mean
            mean = np.array(start_x, dtype=float)
            sigma = sigma0
            
            # Cholesky decomposition update helper
            def update_chol(C, C_sqrt, C_inv_sqrt):
                # Symmetrize C
                C = (C + C.T) / 2.0
                # Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to eigen decomposition if not positive definite
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-12, None)
                    C_sqrt = (vecs * np.sqrt(vals)).T
                    C = (C + C.T) / 2.0
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-12, None)
                    C_inv_sqrt = (vecs * (1.0 / np.sqrt(vals))).T
                # Inverse square root
                try:
                    C_inv_sqrt = np.linalg.cholesky(np.linalg.inv(C))
                    # Note: cholesky returns lower triangular, we need inverse of that
                    # Actually, let's just use inv(C_sqrt).T for simplicity if stable
                    # C_inv_sqrt = np.linalg.inv(C_sqrt).T
                except:
                    # Eigen fallback for inv sqrt
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-12, None)
                    C_inv_sqrt = (vecs * (1.0 / np.sqrt(vals))).T
            
            # Initial setup
            update_chol(C, C_sqrt, C_inv_sqrt)
            
            iters = budget // lam
            if iters < 1:
                iters = 1
                
            for it in range(iters):
                if time.time() > deadline or calls_made >= total_budget:
                    break
                
                # Generate population
                ys = rng.standard_normal((lam, n))
                # Transform using C_sqrt
                # ys are standard normal, so C_sqrt @ y is transformed
                # Note: C_sqrt is lower triangular from cholesky, so we use it directly
                # Actually, standard CMA uses B * D * z where C = B D^2 B^T
                # Here we use Cholesky: C = L L^T. So we sample L z.
                zs = ys @ C_sqrt.T  # Each row is a sample from N(0, C)
                
                ps = mean + sigma * zs
                fs = np.array([wrapper(p) for p in ps])
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ps[order[0]].copy()
                    
                # Update mean
                new_mean = mean + sigma * np.sum(weights[:, None] * ys[:mu], axis=0)
                
                # Update sigma
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
                # Note: C_inv_sqrt @ (new_mean - mean) / sigma
                # (new_mean - mean) / sigma = sum(w_i y_i)
                # So we need C_inv_sqrt @ (sum w_i y_i)
                y_diff = np.sum(weights[:, None] * ys[:mu], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_diff)
                
                # Hessian norm correction
                h_sigma = 1 if np.sqrt(sum(p_sigma**2) / (1 - (1 - c_sigma)**(2*(it+1)))) / np.sqrt(1 - (1 - c_sigma)**(2*(it+1))) < (1.4 + 2/(n+1)) * np.trace(C_inv_sqrt @ C) / n else 0
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ y_diff)
                
                # Update C
                # Delta H = (1/mu_eff) * sum(w_i y_i y_i^T) - (1 - h1) * (1 + c_1 - c_1/mu_eff) * C
                # h1 = 1 if norm(p_c)^2 / (1 - (1-c_c)^(2*(it+1))) < (1 + 2/(n+1)) * (1 - c_c) * (1 + 2/(n+1)) else 0
                # Actually standard h1:
                h1 = 1 if (np.dot(p_c, p_c) / (1 - (1 - c_c)**(2*(it+1)))) < (1 + 2.0/(n+1)) * (1 - c_c) else 0
                
                rank_one = np.outer(y_diff, y_diff)
                rank_mu = np.sum(weights[:, None, None] * np.einsum('ij,ik->ijk', ys[:mu], ys[:mu]), axis=0)
                
                C = (1 + c_1 * (h1 - 1) - c_1/mu_eff) * C + c_1 * (h1 - 1 + 2) * np.outer(p_c, p_c) / mu_eff + c_mu * rank_mu / mu_eff
                
                # Wait, the formula is:
                # C = (1 - c_1) C + c_1 ( (1/mu_eff) * sum w_i y_i y_i^T + (1 - h1) (1 + c_1 - c_1/mu_eff) C )
                # Let's stick to the simpler update:
                # C = (1 - c_1) * C + c_1 * ( rank_mu / mu_eff + (1 - h1) * (1 + c_1 - c_1/mu_eff) * C )
                # This is getting messy. Let's use a simpler, robust update often found in lite implementations.
                
                # Simpler C update:
                # C = (1 - c_1) * C + c_1 * ( (1/mu_eff) * sum(w_i y_i y_i^T) + (1 - h1) * (1 + c_1 - c_1/mu_eff) * C )
                # This requires careful handling.
                
                # Let's restart C update logic to be safe.
                # C = (1 - c_1) * C + c_1 * ( np.outer(p_c, p_c) / mu_eff + rank_mu / mu_eff ) 
                # This is not quite right either.
                
                # Let's use the standard formula:
                # C = (1 - c_1) C + c_1 ( (1/mu_eff) * sum_{i=1}^mu w_i y_i y_i^T + (1 - h1) (1 + c_1 - c_1/mu_eff) C )
                
                term1 = rank_mu / mu_eff
                term2 = (1 - h1) * (1 + c_1 - c_1/mu_eff) * C
                C = (1 - c_1) * C + c_1 * (term1 + term2)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2.0 * n) - 1))
                
                # Update mean
                mean = new_mean
                
                # Update C_sqrt and C_inv_sqrt
                update_chol(C, C_sqrt, C_inv_sqrt)
                
                # Convergence check
                if sigma < 1e-12:
                    break
                    
            return best_x, best_f

        # Strategy:
        # 1. Run CMA-ES with global start (0,0,0,0,0) or random.
        # 2. Use 70% budget for global search.
        # 3. Use 30% budget for local polish starting from best found.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global
        budget_global = int(total_budget * 0.7)
        start_x_global = np.zeros(dim)
        sigma_global = 2.0 # Domain is [-5, 5], so 2 is reasonable
        
        try:
            best_x, best_f = cma_es(start_x_global, sigma_global, budget_global, rng)
        except TimeoutError:
            pass
        except Exception:
            pass
        
        # Phase 2: Local Polish
        if calls_made < total_budget and best_x is not None:
            budget_local = total_budget - calls_made
            # Start from best found, small sigma
            start_x_local = best_x.copy()
            sigma_local = 0.1
            
            try:
                best_x, best_f = cma_es(start_x_local, sigma_local, budget_local, rng)
            except TimeoutError:
                pass
            except Exception:
                pass

        # Final check: if we have very few calls left, just return best
        return best_x, best_f
