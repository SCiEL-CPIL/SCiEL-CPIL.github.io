import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        all_points = []

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim):
            """Generate n Halton sequence points in [0,1)^dim."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    prime = primes[d % len(primes)]
                    f = 1.0
                    ip = i
                    while ip > 0:
                        f /= prime
                        ip = int(ip / prime)
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Deterministic Halton sequence exploration
        n_halton = 20
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts_mapped = -5.0 + 10.0 * halton_pts

        for i in range(n_halton):
            if time.time() > deadline - 20 or calls_made >= budget - 50:
                break
            eval_func(halton_pts_mapped[i])

        # Sort by fitness
        all_points.sort(key=lambda p: p[0])

        # Get top diverse starting points for ES trajectories
        start_points = []
        min_dist = 2.0
        for f, x in all_points:
            if len(start_points) >= 5:
                break
            if all(np.linalg.norm(x - sp) > min_dist for sp in start_points):
                start_points.append(x)

        # Fill if needed
        while len(start_points) < 5 and len(all_points) > len(start_points):
            f, x = all_points[len(start_points)]
            if all(np.linalg.norm(x - sp) > 0.5 for sp in start_points):
                start_points.append(x)
        
        # If still not enough, just use best
        while len(start_points) < 5:
            start_points.append(best_x.copy())

        # Phase 2: Multi-fidelity (1+1)-ES search
        # Define 5 trajectories with different step sizes
        step_sizes = [0.2, 0.5, 1.0, 2.0, 4.0]
        trajectories = []
        for i in range(5):
            trajectories.append({
                'x': start_points[i].copy(),
                'f': func(start_points[i]), # Note: this consumes budget
                'step': step_sizes[i],
                'success': 0,
                'fails': 0
            })
            # Update global best if this start point is better
            if trajectories[-1]['f'] < best_f:
                best_f = trajectories[-1]['f']
                best_x = trajectories[-1]['x'].copy()
            calls_made += 1 # Manual increment because we called func directly to init state
            # Check budget
            if calls_made >= budget:
                break

        # Reserve budget for final NM
        final_nm_budget = max(10, budget // 10)
        es_budget_limit = budget - final_nm_budget

        # Round-robin ES
        while calls_made < es_budget_limit and time.time() < deadline - 10:
            for traj in trajectories:
                if calls_made >= es_budget_limit or time.time() > deadline - 10:
                    break
                
                # Generate candidate
                noise = np.random.randn(dim) * traj['step']
                candidate = traj['x'] + noise
                candidate = np.clip(candidate, -5, 5)
                
                f_cand = eval_func(candidate)
                
                if f_cand < traj['f']:
                    traj['x'] = candidate
                    traj['f'] = f_cand
                    traj['success'] += 1
                    # Increase step slightly if successful
                    traj['step'] = min(5.0, traj['step'] * 1.1)
                else:
                    traj['fails'] += 1
                    # Decrease step if failed
                    traj['step'] = max(0.05, traj['step'] * 0.9)
                
                # Update global best
                if f_cand < best_f:
                    best_f = f_cand
                    best_x = candidate.copy()

        # Phase 3: Final Nelder-Mead refinement
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline - 5:
            try:
                minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining - 5, 100),
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        return best_x, best_f
