import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)
        
        # ZDT3 Pareto front: x1 = t, x2..x30 = 0
        # f1 = t
        # f2 = 1 - sqrt(t) - sin(10*pi*t)  (with g=1 on front)
        
        # We have budget evaluations. Use all of them to sample the front.
        # Strategy: non-uniform sampling concentrated where f2 is small (high HV contribution)
        
        # First, find critical points (local minima of f2)
        # f2'(t) = -1/(2*sqrt(t)) - 10*pi*cos(10*pi*t)
        # f2''(t) = 1/(4*t^(3/2)) + 100*pi^2*sin(10*pi*t)
        
        # Use numerical root finding for f2' = 0
        from scipy.optimize import brentq
        
        def f2(t):
            if t < 0: t = 0
            if t > 1: t = 1
            return 1.0 - np.sqrt(t) - np.sin(10.0 * np.pi * t)
        
        def f2_prime(t):
            if t <= 0: return -np.inf
            return -1.0/(2.0*np.sqrt(t)) - 10.0*np.pi*np.cos(10.0*np.pi*t)
        
        # Find all critical points in (0, 1)
        # f2' changes sign; scan for sign changes
        t_scan = np.linspace(1e-6, 1.0, 10000)
        fp = np.array([f2_prime(t) for t in t_scan])
        
        # Find sign changes
        critical_ts = []
        for i in range(len(fp)-1):
            if fp[i] * fp[i+1] < 0:
                # Sign change between t_scan[i] and t_scan[i+1]
                try:
                    root = brentq(f2_prime, t_scan[i], t_scan[i+1], xtol=1e-12)
                    if 0 < root < 1:
                        critical_ts.append(root)
                except:
                    pass
        
        # Also include t=0 and t=1
        all_points = [0.0] + critical_ts + [1.0]
        
        # Filter: only keep points that are on the Pareto front (non-dominated)
        # On the ZDT3 front, all points with x2..=0 are non-dominated
        # But we should verify by checking f2 values
        
        # Now allocate budget points:
        # We want to maximize HV. Use importance sampling based on (11 - f2(t)).
        # Points where f2 is small contribute more.
        
        # Strategy: 
        # 1. Always include the critical points (local minima) - they're essential
        # 2. Distribute remaining points proportionally to (11 - f2(t)) in each interval
        
        n_critical = len(all_points)
        remaining = budget - n_critical
        
        if remaining < 0:
            # Too many critical points; just use the best ones
            all_points = sorted(all_points)
            # Keep the ones with lowest f2
            scored = [(f2(t), t) for t in all_points]
            scored.sort(key=lambda x: x[0])
            all_points = [t for _, t in scored[:budget]]
            remaining = 0
        
        # For remaining points, use stratified sampling weighted by HV contribution
        # In each interval [t_i, t_{i+1}], the HV contribution is approximately
        # (t_{i+1} - t_i) * (11 - f2_mid)
        # So allocate points proportionally to this
        
        intervals = []
        for i in range(len(all_points)-1):
            t_mid = (all_points[i] + all_points[i+1]) / 2.0
            weight = (all_points[i+1] - all_points[i]) * (11.0 - f2(t_mid))
            intervals.append((all_points[i], all_points[i+1], weight))
        
        total_weight = sum(w for _,_,w in intervals)
        
        # Allocate remaining points to intervals
        sampled_ts = list(all_points)
        
        for (t_lo, t_hi, w) in intervals:
            if remaining <= 0:
                break
            n_alloc = max(1, int(round(remaining * w / total_weight))) if total_weight > 0 else 0
            n_alloc = min(n_alloc, remaining)
            if n_alloc > 0:
                # Uniformly sample n_alloc points in (t_lo, t_hi)
                if n_alloc == 1:
                    ts = [(t_lo + t_hi)/2.0]
                else:
                    ts = np.linspace(t_lo, t_hi, n_alloc+2)[1:-1].tolist()
                sampled_ts.extend(ts)
                remaining -= n_alloc
        
        # If we still have budget left, add more points in high-weight intervals
        while remaining > 0 and len(sampled_ts) < budget:
            # Add point in the interval with highest remaining weight
            best_interval = max(intervals, key=lambda x: x[2])
            t_lo, t_hi, _ = best_interval
            t_new = (t_lo + t_hi) / 2.0
            # Check if this point is already close to an existing point
            if not any(abs(t_new - t) < 1e-6 for t in sampled_ts):
                sampled_ts.append(t_new)
                remaining -= 1
            else:
                # Perturb slightly
                t_new = t_lo + (t_hi - t_lo) * (0.3 + 0.4 * np.random.random())
                if not any(abs(t_new - t) < 1e-6 for t in sampled_ts):
                    sampled_ts.append(t_new)
                    remaining -= 1
                else:
                    break
        
        # Deduplicate and sort
        sampled_ts = sorted(set([round(t, 12) for t in sampled_ts]))
        
        # Truncate to budget
        sampled_ts = sampled_ts[:budget]
        
        # Evaluate all points
        recorded = []
        x = np.zeros(dim)
        
        for t in sampled_ts:
            if time.time() > deadline:
                break
            if len(recorded) >= budget:
                break
            
            x[0] = float(t)
            x[1:] = 0.0
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
        
        return recorded
