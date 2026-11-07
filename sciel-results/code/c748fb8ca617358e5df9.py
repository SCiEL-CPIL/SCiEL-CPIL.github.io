import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        bounds = [(-5.0, 5.0)] * self.dim
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        d = self.dim

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy:
        # 1. Moderate global sampling to find promising basins.
        # 2. Aggressive local refinement using L-BFGS-B (for smooth) and Nelder-Mead (for robustness).
        # 3. Repeated cycles of local search from diverse starts to handle multimodality.
        
        # Phase 1: Global Sampling
        # We want enough samples to cover the space but leave budget for local search.
        # 300 calls total.
        # Let's use ~50-80 for global, rest for local.
        # But local search can be very efficient (10-20 calls for convergence).
        # Let's start with 30 random points.
        
        num_global = 30
        candidates = []
        for i in range(num_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # We will perform local optimization from the top K candidates.
        # K depends on remaining budget.
        # Each local run gets a maxfev limit to ensure we try multiple starts.
        
        # Estimate cost of a local run. L-BFGS-B in 5D usually converges in < 50 evals.
        # NM can be similar or slightly more.
        # Let's set a dynamic maxfev for local runs.
        
        # We'll try to fit as many local runs as possible.
        # Let's say we want at least 5 local runs.
        # If we have 270 calls left, 270/5 = 54 calls per run.
        
        # Let's iterate:
        # Pick the best candidate not yet refined (or just the top K).
        # To avoid redundancy, we can check distance to already refined points,
        # but simple top-K is often sufficient for BBOB.
        
        num_local_runs = 10  # We will try up to 10 local runs
        # But we stop if budget runs out.
        
        # To make it more robust, we'll interleave:
        # Refine top 5, then if budget remains, refine next 5, etc.
        # Actually, just a loop over the sorted candidates is fine.
        
        # We need to be careful not to re-evaluate the same point too much.
        # L-BFGS-B will evaluate the start point again. That's 1 call.
        
        for i in range(num_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Pick a start point
            if i < len(candidates):
                _, x_start = candidates[i]
            else:
                # If we run out of candidates, generate new random ones
                x_start = rng.uniform(-5.0, 5.0, d)
                f = call_func(x_start)
                # Insert into candidates? Not strictly necessary if we just use it.
                # But let's keep the list sorted for consistency if we were to add.
                # For simplicity, just use it.
            
            # Determine maxfev for this local run
            # We want to reserve some budget for future runs.
            # remaining = budget - calls_made
            # runs_left = num_local_runs - i
            # maxfev = remaining // runs_left
            # But min 10, max 50?
            
            remaining = self.budget - calls_made
            runs_left = num_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            maxfev = max(10, min(50, remaining // runs_left))
            
            # Run L-BFGS-B first (good for smooth)
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # If we still have budget and time, polish with Nelder-Mead
            # NM is good for non-smooth or if L-BFGS-B got stuck in a saddle/flat area.
            if calls_made < self.budget and time.time() < deadline:
                # Use a small maxfev for NM polish
                nm_maxfev = max(5, min(20, (self.budget - calls_made) // max(1, runs_left)))
                try:
                    res_nm = minimize(
                        call_func,
                        res.x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except Exception:
                    pass
            
            # If we have lots of budget left (e.g. local converged very fast),
            # we could do another local run from the new best.
            # But the loop handles this by moving to the next candidate.
            # However, if the current run improved best_x significantly,
            # maybe we should restart from there?
            # The candidates list is static from Phase 1.
            # To adapt, we could re-sort or add new points.
            # For now, let's stick to the static list but ensure we use the best_x
            # as a start for subsequent runs if it's better than the next candidate?
            # Actually, if best_x is found, it's likely better than candidates[i+1].
            # So we should start from best_x for the next iteration if it's new.
            
            # Let's modify the loop logic slightly:
            # Instead of just iterating i, let's maintain a priority queue or list of starts.
            # But for simplicity and speed, the static list + L-BFGS-B is often strong enough.
            # Let's add a check: if best_x is significantly better than x_start,
            # we might want to run another local search from best_x immediately.
            # But that's handled by the next iteration if we update candidates.
            
            # Let's update the candidates list with the new best if it's good enough.
            # Actually, just letting the loop continue is fine.
            # If L-BFGS-B converged fast, maxfev was small, so we have budget for more runs.
            # The next iteration will pick candidates[i+1].
            # If candidates[i+1] is worse than the new best_x, we might waste time.
            # Better: always start from the current global best if it's improved?
            # No, that leads to over-refinement of the same basin.
            # We need diversity. The static list provides diversity.
            
            # One improvement: if the local run didn't improve best_f,
            # we might skip NM.
            # We don't easily track "improved" without storing old best_f.
            # Let's just run NM if budget allows.

        # Phase 3: Final Polish
        # If we have budget left, do a final high-precision L-BFGS-B from best_x.
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            # Use a decent chunk for final polish
            final_maxfev = min(remaining, 50)
            if final_maxfev > 5:
                try:
                    res_final = minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': final_maxfev,
                            'ftol': 1e-15,
                            'gtol': 1e-10
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
