import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 280.0  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # 5 trajectories with different step sizes
        num_trajs = 5
        sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
        
        # Initialize trajectories
        trajs = []
        for i in range(num_trajs):
            x_init = rng.uniform(-5, 5, self.dim)
            f_init = func(x_init)
            trajs.append({
                'x': x_init,
                'f': f_init,
                'sigma': sigmas[i],
                'calls': 1,
                'history': [0.0], # Store relative improvements
                'last_improve_call': 1
            })
            
        total_calls = num_trajs
        best_x = trajs[0]['x'].copy()
        best_f = trajs[0]['f']
        
        # Update global best from initial
        for t in trajs:
            if t['f'] < best_f:
                best_f = t['f']
                best_x = t['x'].copy()

        # Helper to calculate merit
        def get_merit(traj):
            h = traj['history']
            if len(h) == 0:
                return 0.0
            recent = h[-20:]
            return np.mean(recent)

        while total_calls < self.budget:
            if time.time() > deadline:
                break
            
            # Select trajectory with highest merit
            merits = [get_merit(t) for t in trajs]
            
            # If all merits are 0 (or very close), prefer the one with best f
            if np.allclose(merits, 0.0):
                best_idx = np.argmin([t['f'] for t in trajs])
            else:
                best_idx = np.argmax(merits)
            
            t = trajs[best_idx]
            
            # Generate candidate
            step = rng.normal(0, 1, self.dim)
            x_new = t['x'] + t['sigma'] * step
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            try:
                f_new = func(x_new)
            except:
                f_new = np.inf
            
            total_calls += 1
            t['calls'] += 1
            
            # Calculate relative improvement
            if f_new < t['f']:
                rel_imp = (t['f'] - f_new) / (abs(t['f']) + 1e-8)
                t['x'] = x_new
                t['f'] = f_new
                t['last_improve_call'] = total_calls
                t['sigma'] = min(t['sigma'] * 1.1, 10.0)
            else:
                rel_imp = 0.0
                t['sigma'] = max(t['sigma'] * 0.9, 0.01)
            
            t['history'].append(rel_imp)
            if len(t['history']) > 50:
                t['history'] = t['history'][-50:]
                
            # Update global best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Restart logic: If no improvement in last 50 calls for this traj, restart
            if total_calls - t['last_improve_call'] > 50:
                if total_calls < self.budget: # Ensure we don't exceed budget
                    t['x'] = best_x + rng.normal(0, 0.5, self.dim)
                    t['x'] = np.clip(t['x'], -5, 5)
                    try:
                        t['f'] = func(t['x'])
                    except:
                        t['f'] = np.inf
                    total_calls += 1
                    t['calls'] += 1
                    t['last_improve_call'] = total_calls
                    t['history'] = [0.0]
                    t['sigma'] = 1.0
                    
                    if t['f'] < best_f:
                        best_f = t['f']
                        best_x = t['x'].copy()

        return best_x, best_f
