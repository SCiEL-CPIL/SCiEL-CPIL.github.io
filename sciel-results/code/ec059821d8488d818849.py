import numpy as np
import time

class Optimizer:
    """
    CMA-ES based optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with diverse starting points.
    2. For the most promising starting points, run CMA-ES with small population.
    3. Use remaining budget for refinement with adaptive (1+1)-ES around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)
        
        # ddPSO Parameters
        N = 10  # Swarm size
        max_iter = budget // N
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max = 1.0
        
        # Initialize Swarm
        particles = []
        for i in range(N):
            x = np.zeros(dim)
            for d in range(dim):
                center = -5.0 + 10.0 * (i + 0.5) / N
                x[d] = center + rng.uniform(-1, 1)
            x = np.clip(x, -5, 5)
            particles.append({
                'x': x,
                'v': np.zeros(dim),
                'pbest_x': x.copy(),
                'pbest_f': float('inf')
            })
            
        # Initial Evaluation
        for i in range(N):
            if not check_time() or calls_made >= budget:
                break
            fx = safe_call(particles[i]['x'])
            if fx is None:
                break
            particles[i]['pbest_f'] = fx
            particles[i]['pbest_x'] = particles[i]['x'].copy()

        # Main Loop
        for it in range(max_iter):
            if not check_time() or calls_made >= budget:
                break
            
            # Update Inertia
            w = w_start + (w_end - w_start) * (it / max(max_iter - 1, 1))
            
            # Update Particles
            for i in range(N):
                if not check_time() or calls_made >= budget:
                    break
                
                p = particles[i]
                
                # Find neighborhood best (ring topology)
                n1 = (i - 1) % N
                n2 = (i + 1) % N
                
                if particles[n1]['pbest_f'] < particles[n2]['pbest_f']:
                    lbest_x = particles[n1]['pbest_x']
                else:
                    lbest_x = particles[n2]['pbest_x']
                
                # Derandomized update: r1=r2=1
                v_new = w * p['v'] + c1 * (p['pbest_x'] - p['x']) + c2 * (lbest_x - p['x'])
                
                # Clip velocity
                v_new = np.clip(v_new, -v_max, v_max)
                
                # Update position
                x_new = p['x'] + v_new
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                fx_new = safe_call(x_new)
                if fx_new is None:
                    break
                
                # Update particle state
                p['v'] = v_new
                p['x'] = x_new
                
                if fx_new < p['pbest_f']:
                    p['pbest_f'] = fx_new
                    p['pbest_x'] = x_new.copy()

        return best_x, best_f
