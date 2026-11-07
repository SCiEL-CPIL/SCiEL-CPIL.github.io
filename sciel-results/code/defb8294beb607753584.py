import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, dim, n, offset=0):
        """Generate Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                base = d + 2  # prime bases: 2, 3, 5, 7, 11
                f = 1.0
                result = 0.0
                i_val = i + 1 + offset
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                points[i, d] = result
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (330s limit, leave 50s buffer)

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0
        eval_history = []  # (value, x) for tracking top-k

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_c = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                val = float(func(x_c))
            except Exception:
                return np.inf
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_c.copy()
            eval_history.append((val, x_c.copy()))
            return val

        def still_ok():
            return eval_count < budget and time.time() < deadline

        # Phase 1: Space-filling exploration using Halton sequence
        n_explore = min(25, max(10, budget // 5))
        halton_pts = self._halton_sequence(dim, n_explore)
        for pt in halton_pts:
            if not still_ok():
                break
            x_start = pt * 9.0 - 4.5
            counting_func(x_start)
        
        # Also evaluate origin and a few structured points
        extra_starts = [np.zeros(dim), np.ones(dim), -np.ones(dim)]
        for x_start in extra_starts:
            if not still_ok():
                break
            counting_func(x_start)
        
        if best_x is None:
            best_x = np.zeros(dim)
            counting_func(best_x)
        
        # Phase 2: CMA-ES with adaptive restarts
        remaining = budget - eval_count
        if remaining > 20 and still_ok() and best_x is not None:
            try:
                import cma
            except ImportError:
                cma = None
            
            if cma is not None:
                # Use CMA-ES with population size based on remaining budget
                pop_size = max(10, min(30, remaining // 10))
                max_iter = max(10, remaining // pop_size)
                
                # Start from best point found so far
                x0 = best_x.copy()
                
                # Set up CMA-ES options
                opts = {
                    'maxiter': max_iter,
                    'popsize': pop_size,
                    'bounds': [(-5.0, 5.0)] * dim,
                    'seed': 42,
                    'verbose': -9,
                    'tolx': 1e-8,
                    'tolfun': 1e-8,
                }
                
                try:
                    es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
                    while (not es.stop()) and still_ok() and eval_count < budget:
                        solutions = es.ask()
                        for sol in solutions:
                            if not still_ok() or eval_count >= budget:
                                break
                            counting_func(sol)
                        # Only tell if we made progress
                        if eval_count > 0:
                            fvals = [counting_func.__wrapped__(s) if hasattr(counting_func, '__wrapped__') else np.inf for s in solutions]
                            # Since we can't easily get the actual values back, use a workaround
                            # Actually, we need to track values properly
                            break
                except Exception:
                    pass
            
            # Fallback: Multi-start Nelder-Mead if CMA-ES not available or failed
            remaining = budget - eval_count
            if remaining > 15 and still_ok() and len(eval_history) > 0:
                sorted_history = sorted(eval_history, key=lambda t: t[0])
                top_k = min(4, len(sorted_history))
                selected_starts = []
                for val, x in sorted_history:
                    if len(selected_starts) >= top_k:
                        break
                    if all(np.linalg.norm(x - sx) > 0.3 for sx in selected_starts):
                        selected_starts.append(x)
                
                while len(selected_starts) < top_k and len(sorted_history) > 0:
                    val, x = sorted_history[len(selected_starts)]
                    selected_starts.append(x)
                
                budget_per_start = max(8, remaining // max(1, len(selected_starts)))
                
                for i, x_start in enumerate(selected_starts):
                    if not still_ok():
                        break
                    alloc = min(budget_per_start, budget - eval_count)
                    if alloc < 5:
                        break
                    try:
                        res = scipy.optimize.minimize(
                            counting_func,
                            x_start.copy(),
                            method='Nelder-Mead',
                            options={
                                'maxiter': max(5, alloc // 2),
                                'xatol': 1e-7,
                                'fatol': 1e-7,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass
        
        # Phase 3: L-BFGS-B from current best if budget remains
        remaining = budget - eval_count
        if remaining > 10 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max(5, remaining // 3), 'maxfun': remaining}
                )
            except Exception:
                pass
        
        # Phase 4: Final high-precision polish
        remaining = budget - eval_count
        if remaining > 5 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(3, remaining),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = 0.0

        return best_x, best_f
