import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Strategy:
        # 1. Initial broad sampling to find a good basin (Schwefel has many minima).
        # 2. Run multiple parallel (1+1)-ES instances from the best initial points.
        #    Each instance uses robust step-size adaptation.
        # 3. If an instance stagnates or its step size becomes too small, restart it
        #    from the current global best with a fresh step size.
        # 4. Round-robin allocation of budget among active instances.

        # Phase 1: Initial Sampling
        # Use 20% of budget for initial exploration if possible, but cap it.
        n_init = min(15, budget // 5)
        
        # Generate diverse initial points
        # 1. Origin
        # 2. Points near known Schwefel minima (4.20968) with noise
        # 3. Random points
        init_points = []
        init_points.append(np.zeros(d))
        for _ in range(3):
            init_points.append(4.20968 + rng.normal(0, 1.0, d))
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-5, 5, d))
        
        init_points = [np.clip(p, -5, 5) for p in init_points]
        
        init_results = []
        for p in init_points:
            try:
                f = safe_func(p)
                init_results.append((f, p.copy()))
            except StopIteration:
                break
        
        if not init_results:
            return best_x, best_f

        # Sort initial results
        init_results.sort(key=lambda x: x[0])
        
        # Select top K points for parallel ES
        K = 5
        top_starts = init_results[:K]
        
        # State for each (1+1)-ES instance
        # x, sigma, f, success_history, stagnation_counter
        es_states = []
        for f, x in top_starts:
            es_states.append({
                'x': x,
                'sigma': 1.0,
                'f': f,
                'success_history': [],
                'stagnation': 0
            })

        if not es_states:
            return best_x, best_f

        # Main Loop: Round-robin
        window_size = 10
        max_stagnation = 50 # Restart if no improvement in 50 steps for this instance
        
        # We will iterate until budget or time is exhausted
        # To ensure fair distribution, we pick the instance with the lowest 'stagnation'
        # or just cycle through them. Cycling is simpler and robust.
        
        idx = 0
        while calls < budget and time.time() < deadline:
            # Cycle through active states
            for state in es_states:
                if calls >= budget or time.time() > deadline:
                    break
                
                x = state['x']
                sigma = state['sigma']
                f_curr = state['f']
                
                # Generate candidate
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                x_candidate = np.clip(x_candidate, -5, 5)
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                
                # Check success
                if f_candidate < f_curr:
                    # Success
                    state['x'] = x_candidate
                    state['f'] = f_candidate
                    state['success_history'].append(True)
                    state['stagnation'] = 0
                else:
                    # Failure
                    state['success_history'].append(False)
                    state['stagnation'] += 1
                
                # Adapt sigma based on success rate in the last `window_size` steps
                if len(state['success_history']) >= window_size:
                    recent = state['success_history'][-window_size:]
                    success_rate = np.mean(recent)
                    
                    # Target success rate ~ 1/ln(d+1) is for standard (1+1)-ES
                    # For d=5, 1/ln(6) ~ 0.61. 
                    # However, with rotation/shift, we might want to be more aggressive.
                    # Let's aim for 0.3 success rate as a balance.
                    
                    if success_rate > 0.4:
                        # Too many successes, increase step size to explore more
                        state['sigma'] *= 1.15
                    elif success_rate < 0.2:
                        # Too many failures, decrease step size
                        state['sigma'] *= 0.85
                    else:
                        # Keep sigma stable
                        pass
                    
                    # Limit sigma
                    state['sigma'] = np.clip(state['sigma'], 1e-6, 10.0)
                    
                    # Trim history
                    if len(state['success_history']) > window_size:
                        state['success_history'].pop(0)
                
                # Stagnation Restart
                # If an instance has been stuck for a long time, restart it from the global best
                # with a larger step size to escape local minima.
                if state['stagnation'] > max_stagnation:
                    # Restart from current global best
                    if best_x is not None:
                        state['x'] = best_x.copy()
                        state['f'] = best_f
                        state['sigma'] = 2.0 # Larger step to explore
                        state['success_history'] = []
                        state['stagnation'] = 0

        return best_x, best_f
