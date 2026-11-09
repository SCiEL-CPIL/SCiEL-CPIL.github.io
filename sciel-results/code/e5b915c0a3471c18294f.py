import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0 # Safety margin from 330s
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1. / np.sum(w**2)
        
        # Strategy parameters
        c1 = (2. / ((dim+1.)**2 + 10.)) * (1. - min(1., (mueff - 2.) / ((dim + 2.)*mueff)))
        cmu = min(1. - c1, 2. * (mueff - 2. + 1./mueff) / ((dim + 2.) * (mueff + 2. - 1./mueff)))
        cs = (mueff + 1. + 2./dim) / (dim + mueff + 5.)
        ds = 1. + 2.*max(0., np.sqrt((mueff-1.)/(dim+1.)) - 1.) + cs
        c_sigma = (cs + 1.) / (dim + cs + 3.)
        
        x_best = None
        f_best = np.inf
        calls_made = 0
        
        # Restart mechanism
        restarts = 0
        max_restarts = 3
        stagnation_limit = 100
        
        while restarts <= max_restarts and calls_made < budget:
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES state
            # Start near 0 or random? Random is safer for multimodal.
            # But 0 is often good for BBOB. Let's use a random point in [-2,2] to be safe.
            rng = np.random.default_rng(42 + restarts)
            x_mean = rng.uniform(-2, 2, dim)
            C = np.eye(dim)
            sigma = 1.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            last_improvement_eval = calls_made
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                # Sample population
                # Compute Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset if C becomes invalid
                    C = np.eye(dim)
                    L = np.eye(dim)
                    sigma = 1.0
                    x_mean = rng.uniform(-5, 5, dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    last_improvement_eval = calls_made
                    stagnation_limit = 50
                    continue

                pop = np.zeros((lam, dim))
                fit = np.zeros(lam)
                
                x_mean_old = x_mean.copy()
                
                for i in range(lam):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(dim)
                    x = x_mean_old + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    
                    fx = func(x)
                    calls_made += 1
                    pop[i] = x
                    fit[i] = fx
                    
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                        last_improvement_eval = calls_made
                
                if calls_made >= budget:
                    break
                    
                # Sort by fitness
                idx = np.argsort(fit)
                pop_sorted = pop[idx]
                fit_sorted = fit[idx]
                
                # Update mean
                x_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
                
                # Update covariance matrix and sigma
                # y_k = (x_k - x_mean_old) / sigma
                Y = (pop_sorted[:mu] - x_mean_old[None, :]) / sigma
                
                # Update ps and pc
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ pc) # This is slightly different, let's use standard
                # Standard: ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (1/sigma) * (x_mean - x_mean_old)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (x_mean - x_mean_old) / sigma
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made - last_improvement_eval)))) < (1.4 + 2. / (dim + 1.)) * np.sqrt((dim + 1.) / (dim - 1.))
                
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_mean - x_mean_old) / sigma
                
                # Update C
                C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * (ps * ps.T - C)) 
                C += cmu * np.sum(w[:, None, None] * Y * Y[:, :, None], axis=0)
                
                # Update sigma
                sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                # Check for stagnation
                if calls_made - last_improvement_eval > stagnation_limit:
                    break
            
            restarts += 1
            
        return x_best, f_best
