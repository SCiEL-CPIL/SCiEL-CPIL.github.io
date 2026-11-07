import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        return self.func(x)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        
        best_x = None
        best_f = float('inf')
        
        # CMA-ES Parameters
        # For dim=5, a population size of 10 is reasonable
        lambda_ = 4 + int(3 * np.log(d))
        if lambda_ < 10:
            lambda_ = 10
            
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        c1 = (2 / ((d + 1.3)**2 + mueff))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff)))
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chi = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Initialization
        # Start at the origin, which is often a good starting point for BBOB functions
        # The function is shifted, but CMA-ES will adapt.
        x_mean = np.zeros(d)
        sigma = 2.0
        
        # Covariance matrix and evolution paths
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute square root of C and inverse square root
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        sqrtC = B @ np.diag(D) @ B.T
        inv_sqrtC = B @ np.diag(1.0/D) @ B.T
        
        # Main Loop
        while self.calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                if self.calls >= self.budget:
                    break
                z = np.random.randn(d)
                x_off = x_mean + sigma * (sqrtC @ z)
                x_off = np.clip(x_off, -5, 5)
                
                f_off = self._evaluate(x_off)
                if f_off is None:
                    break
                offspring.append((f_off, x_off))
                
                # Update global best
                if f_off < best_f:
                    best_f = f_off
                    best_x = x_off.copy()
            
            if not offspring:
                break
                
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x[0])
            
            # If we ran out of budget during generation, just break
            if self.calls >= self.budget:
                break
                
            # Update mean
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += w[i] * offspring[i][1]
            x_mean = new_mean
            
            # Update evolution path for covariance matrix adaptation
            y_vec = inv_sqrtC @ (x_mean - x_mean) # This is wrong, need previous mean
            # Correction: We need the difference between the new weighted mean and the old mean
            # Actually, the standard update is:
            # y = (new_mean - old_mean) / sigma
            # But we overwrote x_mean. Let's store old_mean.
            
            # Let's restructure the loop to be safer and clearer.
            # I will implement a slightly more standard CMA-ES step.
            
            # Break out of this complex inline implementation to avoid state errors
            # and use a cleaner implementation below.
            break
        
        # The above loop had a logic flaw in the mean update. 
        # Let's implement a robust CMA-ES from scratch in the main body to ensure correctness.
        
        # Reset state for a clean run
        self.calls = 0
        best_x = None
        best_f = float('inf')
        
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        sqrtC = B @ np.diag(D) @ B.T
        inv_sqrtC = B @ np.diag(1.0/D) @ B.T
        
        while self.calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                if self.calls >= self.budget:
                    break
                z = np.random.randn(d)
                x_off = x_mean + sigma * (sqrtC @ z)
                x_off = np.clip(x_off, -5, 5)
                
                f_off = self._evaluate(x_off)
                if f_off is None:
                    break
                offspring.append((f_off, x_off))
                
                if f_off < best_f:
                    best_f = f_off
                    best_x = x_off.copy()
            
            if not offspring:
                break
                
            offspring.sort(key=lambda x: x[0])
            
            # If we hit budget exactly, stop
            if self.calls >= self.budget:
                break
                
            # Update mean
            old_mean = x_mean.copy()
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += w[i] * offspring[i][1]
            x_mean = new_mean
            
            # Update evolution paths
            y_vec = (x_mean - old_mean) / sigma
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * y_vec
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrtC @ y_vec)
            
            # Update covariance matrix
            delta = (1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) - cs) * \
                    (np.dot(ps, ps) / d - 1)
            
            C = (1 + c1 * cmu * delta - c1) * C
            for i in range(mu):
                y_i = inv_sqrtC @ (offspring[i][1] - old_mean)
                C += c1 * cmu * w[i] * np.outer(y_i, y_i)
                
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            
            # Re-compute sqrtC and inv_sqrtC periodically or every step
            # For d=5, every step is fine
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            sqrtC = B @ np.diag(D) @ B.T
            inv_sqrtC = B @ np.diag(1.0/D) @ B.T
            
            # Convergence check
            if np.linalg.norm(x_mean - old_mean) < 1e-10:
                break

        # Final Polish: Coordinate Descent on the best found point
        if best_x is not None and self.calls < self.budget and time.time() < deadline - 5:
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1
            
            for _ in range(50):  # Max 50 refinement iterations
                if self.calls >= self.budget or time.time() > deadline - 2:
                    break
                
                improved = False
                for i in range(d):
                    if self.calls >= self.budget:
                        break
                    
                    # Try positive step
                    x_try = x_polish.copy()
                    x_try[i] += step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                        continue
                    
                    # Try negative step
                    x_try = x_polish.copy()
                    x_try[i] -= step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
                
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = self.func(best_x)
        
        return best_x, best_f
