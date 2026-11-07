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

        # Strategy: Parallel multi-scale differential evolution + local refinement
        # Three population scales: large (global), medium (basin ID), small (fine-tuning)
        # Shared archive of best points to avoid redundant evaluations
        
        archive = []  # List of (f, x) tuples, kept sorted by f
        
        def add_to_archive(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Add to archive, keep only top 50 unique points
            archive.append((f, x.copy()))
            archive.sort(key=lambda t: t[0])
            # Remove duplicates (within tolerance) and keep top 50
            unique = []
            for f_i, x_i in archive:
                is_dup = False
                for f_j, x_j in unique:
                    if np.linalg.norm(x_i - x_j) < 0.1:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append((f_i, x_i))
                if len(unique) >= 50:
                    break
            archive.clear()
            archive.extend(unique)

        # Phase 1: Multi-scale DE with three populations
        # Large population for global exploration
        # Medium for basin identification  
        # Small for fine-tuning around best
        
        # Budget allocation: ~60% for DE, ~40% for local refinement
        de_budget = int(self.budget * 0.6)
        local_budget = self.budget - de_budget
        
        # Large population: global exploration
        pop_large = 20
        # Medium population: basin identification
        pop_medium = 10
        # Small population: fine-tuning
        pop_small = 5
        
        # Initialize large population with random points
        pop_l = []
        for i in range(pop_large):
            if calls_made >= de_budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            add_to_archive(x, f)
            pop_l.append((f, x.copy()))
        
        pop_l.sort(key=lambda t: t[0])
        
        # Initialize medium population from top of large + some random
        pop_m = []
        for i in range(min(pop_medium, len(pop_l))):
            f, x = pop_l[i]
            pop_m.append((f, x.copy()))
        while len(pop_m) < pop_medium:
            if calls_made >= de_budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            add_to_archive(x, f)
            pop_m.append((f, x.copy()))
        pop_m.sort(key=lambda t: t[0])
        
        # Initialize small population from top of medium
        pop_s = []
        for i in range(min(pop_small, len(pop_m))):
            f, x = pop_m[i]
            pop_s.append((f, x.copy()))
        pop_s.sort(key=lambda t: t[0])
        
        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate (adaptive)
        
        # Run DE iterations
        de_iter = 0
        max_de_iter = 30
        
        while de_iter < max_de_iter and calls_made < de_budget and time.time() < deadline:
            de_iter += 1
            
            # Adaptive crossover rate: decrease as we converge
            CR_current = max(0.5, 0.9 - 0.02 * de_iter)
            
            # Update large population (global exploration)
            for i in range(len(pop_l)):
                if calls_made >= de_budget or time.time() > deadline:
                    break
                
                # Select three distinct individuals
                idxs = [j for j in range(len(pop_l)) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                
                # DE mutation: v = x_a + F * (x_b - x_c)
                v = pop_l[a][1] + F * (pop_l[b][1] - pop_l[c][1])
                v = np.clip(v, lb, ub)
                
                # Crossover
                j_rand = rng.integers(0, d)
                trial = np.copy(pop_l[i][1])
                for j in range(d):
                    if rng.random() < CR_current or j == j_rand:
                        trial[j] = v[j]
                
                f_trial = call_func(trial)
                add_to_archive(trial, f_trial)
                
                # Selection
                if f_trial <= pop_l[i][0]:
                    pop_l[i] = (f_trial, trial.copy())
            
            pop_l.sort(key=lambda t: t[0])
            
            # Update medium population (basin identification)
            for i in range(len(pop_m)):
                if calls_made >= de_budget or time.time() > deadline:
                    break
                
                idxs = [j for j in range(len(pop_m)) if j != i]
                if len(idxs) < 3:
                    # Add from large population if needed
                    extra = [pop_l[j] for j in range(min(3, len(pop_l))) if pop_l[j] not in pop_m]
                    if len(extra) >= 3:
                        a, b, c = extra[0][1], extra[1][1], extra[2][1]
                        v = a + F * (b - c)
                    else:
                        continue
                else:
                    a_idx, b_idx, c_idx = rng.choice(idxs, size=3, replace=False)
                    a, b, c = pop_m[a_idx][1], pop_m[b_idx][1], pop_m[c_idx][1]
                
                v = a + F * (b - c)
                v = np.clip(v, lb, ub)
                
                j_rand = rng.integers(0, d)
                trial = np.copy(pop_m[i][1])
                for j in range(d):
                    if rng.random() < CR_current or j == j_rand:
                        trial[j] = v[j]
                
                f_trial = call_func(trial)
                add_to_archive(trial, f_trial)
                
                if f_trial <= pop_m[i][0]:
                    pop_m[i] = (f_trial, trial.copy())
            
            pop_m.sort(key=lambda t: t[0])
            
            # Update small population (fine-tuning)
            for i in range(len(pop_s)):
                if calls_made >= de_budget or time.time() > deadline:
                    break
                
                # Use smaller F for fine-tuning
                F_small = 0.5
                
                idxs = [j for j in range(len(pop_s)) if j != i]
                if len(idxs) < 3:
                    # Use archive points
                    if len(archive) >= 3:
                        a = archive[0][1]
                        b = archive[1][1]
                        c = archive[2][1]
                        v = a + F_small * (b - c)
                    else:
                        continue
                else:
                    a_idx, b_idx, c_idx = rng.choice(idxs, size=3, replace=False)
                    a = pop_s[a_idx][1]
                    b = pop_s[b_idx][1]
                    c = pop_s[c_idx][1]
                    v = a + F_small * (b - c)
                
                v = np.clip(v, lb, ub)
                
                j_rand = rng.integers(0, d)
                trial = np.copy(pop_s[i][1])
                for j in range(d):
                    if rng.random() < CR_current or j == j_rand:
                        trial[j] = v[j]
                
                f_trial = call_func(trial)
                add_to_archive(trial, f_trial)
                
                if f_trial <= pop_s[i][0]:
                    pop_s[i] = (f_trial, trial.copy())
            
            pop_s.sort(key=lambda t: t[0])
            
            # Sync populations: promote best from small to medium, medium to large
            # Keep top 5 of small in medium
            for i in range(min(5, len(pop_s))):
                if pop_s[i][0] < pop_m[-1][0]:
                    pop_m[-1] = pop_s[i]
            pop_m.sort(key=lambda t: t[0])
            
            # Keep top 10 of medium in large
            for i in range(min(10, len(pop_m))):
                if pop_m[i][0] < pop_l[-1][0]:
                    pop_l[-1] = pop_m[i]
            pop_l.sort(key=lambda t: t[0])
        
        # Phase 2: Local refinement from top archive points
        # Use remaining budget for local search
        
        # Get top K unique points from archive
        top_k = min(10, len(archive))
        starts = []
        for f, x in archive:
            is_dup = False
            for _, x_prev in starts:
                if np.linalg.norm(x - x_prev) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                starts.append((f, x))
            if len(starts) >= top_k:
                break
        
        # If not enough diverse starts, add random
        while len(starts) < 5 and calls_made < self.budget and time.time() < deadline:
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            starts.append((f, x))
        
        starts.sort(key=lambda t: t[0])
        
        # Local refinement
        num_local_runs = min(len(starts), 8)
        
        for i in range(num_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            _, x_start = starts[i]
            
            remaining = self.budget - calls_made
            runs_left = num_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            maxfev = max(10, min(40, remaining // runs_left))
            
            # L-BFGS-B for smooth functions
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
            
            # Nelder-Mead polish for robustness
            if calls_made < self.budget and time.time() < deadline:
                nm_maxfev = max(5, min(15, (self.budget - calls_made) // max(1, runs_left)))
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
        
        # Phase 3: Final high-precision polish
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            final_maxfev = min(remaining, 30)
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
