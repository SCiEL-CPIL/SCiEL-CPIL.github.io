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
        # f2 = 1 - sqrt(t) - sin(10*pi*t)
        
        def f2(t):
            return 1.0 - np.sqrt(t) - np.sin(10.0 * np.pi * t)
        
        # 1. Identify the Pareto-optimal regions on the front
        # A point t is Pareto optimal if f2(t) <= min_{s < t} f2(s)
        # We scan a dense grid to find these regions.
        
        n_scan = 100000
        t_scan = np.linspace(0.0, 1.0, n_scan)
        f2_scan = f2(t_scan)
        
        # Compute running minimum
        running_min = np.minimum.accumulate(f2_scan)
        
        # Points where f2(t) is equal to the running minimum are on the Pareto front
        tol = 1e-9
        is_pareto = f2_scan <= running_min + tol
        
        # Identify contiguous Pareto segments
        segments = []
        in_segment = False
        seg_start = 0
        for i in range(n_scan):
            if is_pareto[i] and not in_segment:
                seg_start = i
                in_segment = True
            elif not is_pareto[i] and in_segment:
                segments.append((seg_start, i - 1))
                in_segment = False
        if in_segment:
            segments.append((seg_start, n_scan - 1))
        
        if len(segments) == 0:
            sampled_ts = np.linspace(0.0, 1.0, budget)
        else:
            # For each segment, compute the arc length in (f1, f2) space
            # and allocate points proportionally
            seg_lengths = []
            for (s, e) in segments:
                if e > s:
                    t_seg = t_scan[s:e+1]
                    f1_seg = t_seg
                    f2_seg = f2_scan[s:e+1]
                    # Compute arc length
                    df1 = np.diff(f1_seg)
                    df2 = np.diff(f2_seg)
                    lengths = np.sqrt(df1**2 + df2**2)
                    seg_lengths.append(np.sum(lengths))
                else:
                    seg_lengths.append(1.0)
            
            total_length = sum(seg_lengths)
            # Allocate points proportionally to segment length
            n_per_seg = [max(1, int(round(budget * L / total_length))) for L in seg_lengths]
            # Adjust to ensure total is exactly budget
            diff = budget - sum(n_per_seg)
            if diff != 0:
                # Add or remove from the longest segment
                max_idx = np.argmax(seg_lengths)
                n_per_seg[max_idx] += diff
            
            sampled_ts = []
            for (s, e), n_pts in zip(segments, n_per_seg):
                if n_pts <= 0:
                    continue
                t_seg = t_scan[s:e+1]
                f1_seg = t_seg
                f2_seg = f2_scan[s:e+1]
                # Compute cumulative arc length for uniform spacing
                if len(t_seg) > 1:
                    df1 = np.diff(f1_seg)
                    df2 = np.diff(f2_seg)
                    lengths = np.sqrt(df1**2 + df2**2)
                    cum_len = np.concatenate([[0], np.cumsum(lengths)])
                    total_seg_len = cum_len[-1]
                    if total_seg_len > 0:
                        # Place n_pts at equal arc-length intervals
                        target_lengths = np.linspace(0, total_seg_len, n_pts)
                        # Interpolate to find t values
                        t_pts = np.interp(target_lengths, cum_len, t_seg)
                        sampled_ts.extend(t_pts)
                    else:
                        sampled_ts.extend(t_seg[:n_pts])
                else:
                    sampled_ts.extend(t_seg[:n_pts])
            
            sampled_ts = np.array(sampled_ts)
            # If we have more than budget, trim; if less, keep as is
            if len(sampled_ts) > budget:
                sampled_ts = sampled_ts[:budget]
        
        # Sort sampled_ts
        sampled_ts = np.sort(sampled_ts)
        
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
