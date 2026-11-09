import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0 # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi
        
        rng = np.random.default_rng(42)
        random.seed(42)
        
        # Phase 1: Coarse Global Search (Scout)
        scout_budget = min(50, max(20, budget // 10))
        
        best_x = None
        best_f = np.inf
        scout_history = []
        
        # Sample uniformly
        for _ in range(scout_budget):
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_history.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            
        # Sort scout history to find distinct basins
        scout_history.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Local Search with CMA-ES-like adaptation
        K = 5
        starts = []
        threshold = 1.5 # Larger threshold to ensure diversity
        
        for fx, x in scout_history:
            is_distinct = True
            for sx in starts:
                if np.linalg.norm(x - sx) < threshold:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x)
            if len(starts) >= K:
                break
                
        while len(starts) < K:
            starts.append(best_x + rng.normal(0, 0.5, d))
            
        remaining_budget = budget - scout_budget
        budget_per_start = max(10, remaining_budget // K)
        
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        for start_idx, start_x in enumerate(starts):
            if time.time() - self.start_time > self.deadline:
                break
                
            m = start_x.copy()
            sigma = 1.0
            C = np.eye(d)
            step_size = sigma
            
            # Get f value for start
            m_f = None
            for fx, sx in scout_history:
                if np.allclose(sx, start_x):
                    m_f = fx
                    break
            if m_f is None:
                m_f = func(m)
                
            if m_f < global_best_f:
                global_best_f = m_f
                global_best_x = m.copy()
                
            current_budget = budget_per_start
            no_improve = 0
            max_no_improve = 15
            
            while current_budget > 0:
                if time.time() - self.start_time > self.deadline:
                    break
                    
                # Generate candidate using current covariance
                y = rng.multivariate_normal(np.zeros(d), C)
                x_new = m + step_size * y
                x_new = np.clip(x_new, lo, hi)
                
                f_new = func(x_new)
                current_budget -= 1
                
                if f_new < m_f:
                    # Update mean and covariance
                    old_m = m.copy()
                    m = x_new.copy()
                    m_f = f_new
                    no_improve = 0
                    
                    # Update covariance (simplified CMA-ES update)
                    diff = x_new - old_m
                    rank1 = np.outer(diff, diff)
                    C = 0.9 * C + 0.1 * rank1
                    
                    # Normalize C to keep trace reasonable
                    trace = np.trace(C)
                    if trace > 0:
                        C = C / trace * d
                        
                    # Slightly reduce step size on success
                    step_size *= 0.95
                else:
                    no_improve += 1
                    # If stuck, increase step size and reset covariance
                    if no_improve >= max_no_improve:
                        step_size *= 1.5
                        C = np.eye(d)
                        no_improve = 0
                    
                    # Occasionally add noise to covariance to escape local minima
                    if no_improve % 5 == 0:
                        C += 0.1 * np.eye(d)
                        trace = np.trace(C)
                        if trace > 0:
                            C = C / trace * d
                
                if m_f < global_best_f:
                    global_best_f = m_f
                    global_best_x = m.copy()
                    
                if step_size < 0.01:
                    break
                if step_size > 5.0:
                    step_size = 5.0

        return global_best_x, global_best_f
