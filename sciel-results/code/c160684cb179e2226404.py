"""Combined program: Multi-start Nelder-Mead with dynamic budget allocation for Schwefel f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer:
    1. Initialize 10 diverse starting points.
    2. Run Nelder-Mead on each, allocating remaining budget dynamically 
       based on improvement rate (convergence speed).
    3. This shifts from stochastic exploration to deterministic local refinement
       of multiple basins, better suited for deceptive multimodal landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.shape != (n,):
                x_arr = x_arr.reshape(-1)
                if x_arr.size != n:
                    raise ValueError("Wrong dimension")
            f_val = float(func(x_arr))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Create 10 diverse starting points
        num_starts = 10
        starts = []
        for i in range(num_starts):
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-3, 3, n)
            else:
                x = rng.uniform(-4.5, 4.5, n)
            try:
                fx = safe_func(x)
            except (TimeoutError, StopIteration):
                break
            starts.append({
                'x': x.copy(),
                'f': fx,
                'prev_f': fx,
                'calls_used': 1,
                'improvement': 0.0,
                'active': True
            })
        
        if not starts:
            return best_x, best_f
        
        # Sort starts by initial function value
        starts.sort(key=lambda s: s['f'])
        
        # Allocate remaining budget dynamically among active starts
        # Each start gets a share based on its improvement rate
        remaining_budget = self.budget - calls
        
        # We'll iterate, giving each active start a chunk of evaluations
        # and adjusting their priority based on improvement
        eval_chunk = 15  # evaluations per round per start
        
        while calls < self.budget and time.time() < deadline:
            # Identify active starts
            active_starts = [s for s in starts if s['active']]
            
            if not active_starts:
                break
            
            # Compute priority scores based on recent improvement
            # Higher improvement rate = higher priority
            for s in active_starts:
                # improvement = how much it improved since last round
                # Use relative improvement to handle different scales
                if s['prev_f'] > 0:
                    s['improvement'] = max(0, (s['prev_f'] - s['f']) / max(s['prev_f'], 1e-8))
                else:
                    s['improvement'] = max(0, s['prev_f'] - s['f'])
            
            # Sort by improvement rate (descending)
            active_starts.sort(key=lambda s: s['improvement'], reverse=True)
            
            # Allocate evaluations: give more to better-improving starts
            # Use a softmax-like allocation based on improvement
            improvements = np.array([max(s['improvement'], 1e-6) for s in active_starts])
            # Temperature to control concentration
            temp = 0.5
            weights = np.exp(improvements / temp)
            weights /= weights.sum()
            
            # Allocate remaining budget proportionally, but at least 1 eval per active start
            min_evals = len(active_starts)
            if remaining_budget < min_evals:
                break
            
            # Distribute remaining budget
            alloc = np.maximum(1, (weights * (remaining_budget - min_evals + min_evals)).astype(int))
            # Ensure total doesn't exceed remaining
            total_alloc = alloc.sum()
            if total_alloc > remaining_budget:
                # Scale down
                scale = remaining_budget / total_alloc
                alloc = np.maximum(1, (alloc * scale).astype(int))
                # Adjust to exactly match remaining
                diff = remaining_budget - alloc.sum()
                if diff > 0:
                    alloc[0] += diff
                elif diff < 0:
                    # Remove from largest
                    idx = np.argmax(alloc)
                    alloc[idx] += diff
            
            # Run Nelder-Mead on each active start with allocated budget
            for i, s in enumerate(active_starts):
                if alloc[i] <= 0 or calls >= self.budget or time.time() > deadline:
                    continue
                
                # Update previous f for improvement tracking
                s['prev_f'] = s['f']
                
                try:
                    result = scipy.optimize.minimize(
                        safe_func,
                        s['x'],
                        method='Nelder-Mead',
                        options={
                            'maxiter': alloc[i],
                            'xatol': 1e-5,
                            'fatol': 1e-5,
                            'adaptive': True
                        }
                    )
                    s['x'] = result.x
                    s['f'] = result.fun
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    # Keep current state if Nelder-Mead fails
                    pass
                
                # Deactivate start if it hasn't improved much or used all its allocation
                # Simple heuristic: if improvement is very small, deactivate
                if s['improvement'] < 1e-4 and calls > self.budget * 0.5:
                    s['active'] = False
            
            # Update remaining budget
            remaining_budget = self.budget - calls
        
        return best_x, best_f
# EVOLVE-BLOCK-END
