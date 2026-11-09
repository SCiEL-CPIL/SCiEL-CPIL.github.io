"""Restartable Multi-Start CMA-ES for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        total_budget = self.budget
        
        # Global state
        best_x_global = np.zeros(d)
        best_f_global = float('inf')
        global_calls = 0
        
        # CMA-ES Parameters (Standard for n=5)
        lambda_pop = 8
        mu = lambda_pop // 2
        
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 2)) + c_sigma
        
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Strategy: Run multiple independent CMA-ES instances.
        # We will manage a list of 'active' optimizers.
        # Each optimizer has its own state.
        # We allocate budget dynamically.
        
        class CMAInstance:
            def __init__(self, x0, sigma0):
                self.x_mean = x0.copy()
                self.sigma = sigma0
                self.C = np.eye(d)
                self.p_sigma = np.zeros(d)
                self.p_c = np.zeros(d)
                self.B = np.linalg.cholesky(self.C)
                self.D = np.prod(np.sqrt(np.diag(self.C)))
                self.calls_made = 0
                self.best_x_local = x0.copy()
                self.best_f_local = float('inf')
                self.last_improvement_call = 0
                self.stagnation_counter = 0
                self.active = True
                self.generation = 0

            def step(self, func):
                nonlocal global_calls, best_x_global, best_f_global
                
                if not self.active:
                    return
                
                # Check time
                if time.time() > deadline:
                    self.active = False
                    return
                
                # Generate population
                y_list = []
                x_list = []
                f_list = []
                
                for i in range(lambda_pop):
                    if global_calls >= total_budget:
                        self.active = False
                        return
                    if time.time() > deadline:
                        self.active = False
                        return
                        
                    z = np.random.standard_normal(d)
                    y = self.B @ z
                    x = self.x_mean + self.sigma * y
                    x = np.clip(x, -5, 5)
                    
                    fx = func(x)
                    global_calls += 1
                    self.calls_made += 1
                    
                    y_list.append(y)
                    x_list.append(x)
                    f_list.append(fx)
                    
                    if fx < self.best_f_local:
                        self.best_f_local = fx
                        self.best_x_local = x.copy()
                        self.last_improvement_call = global_calls
                        
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = x.copy()
                
                if not f_list:
                    return
                    
                # Update internal state
                idx = np.argsort(f_list)
                y_sorted = np.array([y_list[i] for i in idx[:mu]])
                x_sorted = np.array([x_list[i] for i in idx[:mu]])
                
                self.x_mean = np.dot(weights, x_sorted)
                
                y_w = np.dot(weights, y_sorted)
                
                # h_sigma calculation
                norm_y_w = np.linalg.norm(y_w)
                threshold = (1.4 + 2.0 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
                if norm_y_w / np.sqrt(1 - (1 - np.exp(-1.0 / d_sigma))**2) < threshold:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
                    
                self.p_sigma = (1 - c_sigma) * self.p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
                self.p_c = (1 - c_c) * self.p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                
                delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(self.p_c, self.p_c)
                
                self.C = (1 + c_1 + c_mu - c_1 - c_mu) * self.C \
                    + c_1 * (np.outer(self.p_c, self.p_c) - self.C - delta_h) \
                    + c_mu * np.sum([w * np.outer(y, y) for w, y in zip(weights, y_sorted)], axis=0)
                    
                self.C = (self.C + self.C.T) / 2.0
                
                # Regularize
                try:
                    eigvals = np.linalg.eigvalsh(self.C)
                    if np.min(eigvals) < 1e-12:
                        self.C += (1e-12 - np.min(eigvals)) * np.eye(d)
                except:
                    self.C += 1e-12 * np.eye(d)
                
                try:
                    self.B = np.linalg.cholesky(self.C)
                except np.linalg.LinAlgError:
                    self.C = np.eye(d)
                    self.B = np.eye(d)
                    
                self.D = np.prod(np.sqrt(np.diag(self.C)))
                
                self.sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(self.p_sigma) / np.sqrt(mu_eff) - 1.0))
                self.sigma = np.clip(self.sigma, 1e-12, 5.0)
                
                self.generation += 1
                
                # Stagnation check: if no improvement in last 50 calls, mark for potential restart
                if global_calls - self.last_improvement_call > 50:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

        # Initialize first instance
        instances = []
        initial_x = np.zeros(d)
        initial_f = func(initial_x)
        global_calls += 1
        best_f_global = initial_f
        best_x_global = initial_x.copy()
        
        inst0 = CMAInstance(initial_x, 1.0)
        inst0.best_f_local = initial_f
        inst0.best_x_local = initial_x.copy()
        inst0.last_improvement_call = global_calls
        instances.append(inst0)
        
        # Restart strategy parameters
        max_active = 3
        restart_threshold_stag = 5 # generations of stagnation? No, use call count.
        
        while global_calls < total_budget:
            if time.time() > deadline:
                break
                
            # 1. Step all active instances
            for inst in instances:
                if not inst.active:
                    continue
                inst.step(func)
                if global_calls >= total_budget or time.time() > deadline:
                    break
            
            if global_calls >= total_budget or time.time() > deadline:
                break
            
            # 2. Cleanup and Restart Logic
            # Remove inactive or stagnated instances
            new_instances = []
            for inst in instances:
                if not inst.active:
                    continue
                # If stagnation is high, kill it
                if inst.stagnation_counter > 10: # ~10 generations without improvement
                    continue
                new_instances.append(inst)
            
            instances = new_instances
            
            # If we have capacity and budget, spawn new instances
            # Spawn from global best with perturbed sigma, or random location
            while len(instances) < max_active and global_calls < total_budget:
                # 50% chance to start from global best, 50% random
                if np.random.rand() < 0.5 and best_f_global < float('inf'):
                    x_start = best_x_global + np.random.normal(0, 0.5, d)
                    x_start = np.clip(x_start, -5, 5)
                    sigma_start = 0.5
                else:
                    x_start = np.random.uniform(-4, 4, d)
                    sigma_start = 1.0
                
                new_inst = CMAInstance(x_start, sigma_start)
                # Evaluate starting point to initialize
                if global_calls < total_budget:
                    f_start = func(x_start)
                    global_calls += 1
                    new_inst.best_f_local = f_start
                    new_inst.best_x_local = x_start.copy()
                    new_inst.last_improvement_call = global_calls
                    if f_start < best_f_global:
                        best_f_global = f_start
                        best_x_global = x_start.copy()
                
                instances.append(new_inst)
                
                if time.time() > deadline:
                    break
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
