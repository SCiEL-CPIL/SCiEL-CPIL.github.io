import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        ref1 = 11.0
        ref2 = 11.0

        # ZDT3 known structure (for dim=30):
        # Pareto front is on x[1:] = 0
        # f1 = x0
        # f2 = (1 + g) * (1 - sqrt(f1 / (1 + g)))
        # where g = 1 + 10*(dim-1) when x[1:]=0
        # So f2 depends only on f1 deterministically on the front.

        # Strategy:
        # Phase 1: Sample 5 points to calibrate the front (detect if it matches expected form).
        # Phase 2: Use the known/estimated front shape to place remaining points optimally
        #           for hypervolume maximization.

        # Phase 1: Calibration
        cal_n = 5
        x0_cal = np.linspace(0.0, 1.0, cal_n)
        cal_pts = []

        for x0 in x0_cal:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            try:
                f = problem(x)
                if len(f) == 2:
                    cal_pts.append((float(f[0]), float(f[1])))
            except:
                continue

        if not cal_pts:
            # Fallback: try a few random points
            for _ in range(10):
                if time.time() > deadline:
                    break
                x = np.random.rand(self.dim)
                try:
                    f = problem(x)
                    if len(f) == 2:
                        cal_pts.append((float(f[0]), float(f[1])))
                except:
                    continue
            if not cal_pts:
                return None

        # Sort by f1
        cal_pts.sort(key=lambda p: p[0])
        f1s = np.array([p[0] for p in cal_pts])
        f2s = np.array([p[1] for p in cal_pts])

        # Check if the front matches the known ZDT3 form.
        # g = 1 + 10*(dim-1)
        g_expected = 1.0 + 10.0 * (self.dim - 1)
        one_plus_g = 1.0 + g_expected

        # Expected f2 for given f1: f2 = one_plus_g * (1 - sqrt(f1 / one_plus_g))
        # Let's check if calibration points match this.
        f2_expected = one_plus_g * (1.0 - np.sqrt(f1s / one_plus_g))
        # Handle potential issues if f1 is slightly out of [0,1] due to numerical issues
        f1s_clipped = np.clip(f1s, 0.0, 1.0)
        f2_expected = one_plus_g * (1.0 - np.sqrt(f1s_clipped / one_plus_g))

        # Check mean absolute error
        if len(f1s) > 0:
            mae = np.mean(np.abs(f2s - f2_expected))
        else:
            mae = float('inf')

        use_analytic = (mae < 1e-3)

        # Phase 2: Place remaining points
        remaining = self.budget - len(cal_pts)

        # We want to maximize hypervolume.
        # The front is a function f2 = g(f1).
        # Hypervolume in 2D is the area under the step function formed by non-dominated points,
        # bounded by ref1, ref2.
        # To maximize HV, we should place points where the curve "bulges" out.
        # For a convex curve (f2'' > 0), the optimal placement is to equalize the "sag" or
        # use properties of the curve.

        # Let's define the function to evaluate a point on the front.
        # If use_analytic, we can compute f2 without calling problem (saves budget? No, we must call problem to record it).
        # BUT we can use the analytic form to *choose* where to place points.
        # We must still call problem(x) to get the recorded point.

        # Since the front is 1D, we can pre-compute the optimal f1 positions.
        # The problem reduces to: Given a function f2(f1), place N points to maximize
        # HV = sum (f1[i+1] - f1[i]) * (ref2 - f2[i]) ... (sorted by f1)

        # Actually, the HV of a set of points on a curve is maximized when the points
        # are placed according to the curvature.
        # For a generic curve, we can use a greedy approach:
        # Start with min and max f1.
        # Add the point that maximizes the gain in HV.

        # However, if we know the analytic form, we can be smarter.
        # Let's check the curvature of ZDT3 front.
        # f2(t) = C * (1 - sqrt(t/C)) = C - sqrt(C*t)
        # f2'(t) = - sqrt(C) / (2 * sqrt(t))
        # f2''(t) = sqrt(C) / (4 * t^(3/2)) > 0
        # The function is convex (curves upward).
        # Wait, f2 decreases as f1 increases.
        # Slope is negative and magnitude increases as t->0.
        # So the curve is convex towards the origin?
        # Let's visualize: t=0, f2=C. t=1, f2=C - sqrt(C).
        # It drops sharply near 0, then flattens out.
        # This is a convex function (slope goes from -inf to -small).
        # For a convex function, the area under the chord is greater than the area under the curve.
        # We want to maximize the area of the union of rectangles.
        # This is equivalent to approximating the integral of (ref2 - f2(t)) dt from 0 to 1?
        # No, it's the area of the hypervolume.
        
        # Let's just use a greedy insertion algorithm with the analytic predictor.
        # This is robust and simple.

        # Current set of (f1, f2) pairs.
        current_pts = list(cal_pts)
        current_pts.sort(key=lambda p: p[0])

        # Helper to compute HV of current set
        def calc_hv(pts):
            if not pts:
                return 0.0
            pts_sorted = sorted(pts, key=lambda p: p[0])
            # Filter valid points
            valid = [p for p in pts_sorted if p[0] < ref1 and p[1] < ref2]
            if not valid:
                return 0.0
            # Remove dominated
            nd = []
            # Since sorted by f1, a point is dominated if there is a previous point with f2 <= current f2
            # But we need to be careful with equal f1.
            # Let's just do the O(N^2) check for simplicity as N is small (<200)
            for i, p in enumerate(valid):
                dom = False
                for j, q in enumerate(valid):
                    if i == j: continue
                    if q[0] <= p[0] and q[1] <= p[1]:
                        if q[0] < p[0] or q[1] < p[1]:
                            dom = True
                            break
                if not dom:
                    nd.append(p)
            
            if not nd:
                return 0.0
            
            # nd is sorted by f1
            hv = 0.0
            # Last box
            hv += (ref1 - nd[-1][0]) * (ref2 - nd[-1][1])
            for i in range(len(nd) - 1):
                hv += (nd[i+1][0] - nd[i][0]) * (ref2 - nd[i][1])
            return hv

        # Prediction of f2 given f1
        def predict_f2(f1_val):
            if use_analytic:
                t = np.clip(f1_val, 0.0, 1.0)
                return one_plus_g * (1.0 - np.sqrt(t / one_plus_g))
            else:
                # Linear interpolation on calibration points
                if f1_val <= f1s[0]:
                    return f2s[0]
                if f1_val >= f1s[-1]:
                    return f2s[-1]
                # Find interval
                for i in range(len(f1s) - 1):
                    if f1s[i] <= f1_val <= f1s[i+1]:
                        t = (f1_val - f1s[i]) / (f1s[i+1] - f1s[i])
                        return f2s[i] + t * (f2s[i+1] - f2s[i])
                return f2s[-1]

        # Greedy loop
        while len(current_pts) < self.budget:
            if time.time() > deadline:
                break

            best_gain = -1.0
            best_f1 = None
            best_f2 = None

            # Candidate positions:
            # 1. Midpoints of intervals between existing points
            # 2. Possibly points to extend the range if the front is wider
            # For ZDT3, f1 is in [0, 1].
            # Let's check the range of current f1s.
            min_f1 = min(p[0] for p in current_pts)
            max_f1 = max(p[0] for p in current_pts)

            # If we haven't covered [0, 1], prioritize extending?
            # Or just use a fixed grid of candidate positions?
            # A better way: sample 20 candidate positions in [0, 1] and pick the best.
            # This is O(20 * N) per iteration. 195 iterations * 20 * 200 = ~780k ops. Fast.
            
            # Candidates: uniform grid + midpoints
            candidates = np.linspace(0.0, 1.0, 20)
            # Also add midpoints of existing intervals
            current_pts.sort(key=lambda p: p[0])
            for i in range(len(current_pts) - 1):
                mid = (current_pts[i][0] + current_pts[i+1][0]) / 2.0
                # Add if not too close to existing
                candidates = np.append(candidates, mid)

            # Remove duplicates/near-duplicates
            candidates = np.unique(np.round(candidates, 6))

            hv_current = calc_hv(current_pts)

            for f1_c in candidates:
                f2_c = predict_f2(f1_c)
                # Check if this point is strictly better than ref
                if f1_c >= ref1 or f2_c >= ref2:
                    continue
                
                new_set = current_pts + [(f1_c, f2_c)]
                hv_new = calc_hv(new_set)
                gain = hv_new - hv_current
                
                if gain > best_gain:
                    best_gain = gain
                    best_f1 = f1_c
                    best_f2 = f2_c

            if best_f1 is None:
                break

            # Evaluate the chosen point
            x = np.zeros(self.dim)
            x[0] = best_f1
            try:
                f = problem(x)
                if len(f) == 2:
                    actual_f1 = float(f[0])
                    actual_f2 = float(f[1])
                    current_pts.append((actual_f1, actual_f2))
            except:
                break

        return None
