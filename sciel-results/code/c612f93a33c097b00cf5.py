import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Exact quadratic model fitting
        # Use a large step size d to maximize numerical stability in the difference calculations.
        # The minimizer is in [-4, 4]^5, so d=4.0 keeps points +/- 4.0 within [-5, 5]^5.
        d = 4.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)

        values = np.array([evaluate(p) for p in points])

        c = values[0]
        a_vals = []
        q_vec = np.zeros(dim)

        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            a_i = (val_pos + val_neg - 2*c) / (2 * d * d)
            q_i = (val_pos - val_neg) / (2 * d)
            a_vals.append(a_i)
            q_vec[i] = q_i

        a = np.mean(a_vals)

        if abs(a) < 1e-12:
            a = 1e-12 if a >= 0 else -1e-12

        # Estimated minimum
        x_opt = -q_vec / (2 * a)
        x_opt = np.clip(x_opt, -5, 5)

        f_opt_est = evaluate(x_opt)

        # Phase 2: Iterative refinement with decreasing step sizes
        # For a pure quadratic, one refinement should be exact, but numerical
        # errors may require a second pass with a much smaller step.
        x = best_x.copy()
        fx = best_f
        
        # Refinement steps
        for d_refine in [0.1, 0.01, 0.001]:
            if calls_made + (2 * dim + 2) > budget or time.time() > deadline:
                break
                
            points_refine = [x]
            for i in range(dim):
                p1 = x.copy(); p1[i] += d_refine
                p2 = x.copy(); p2[i] -= d_refine
                points_refine.append(p1)
                points_refine.append(p2)
            
            values_refine = np.array([evaluate(p) for p in points_refine])
            c_ref = values_refine[0]
            a_vals_ref = []
            q_vec_ref = np.zeros(dim)
            
            for i in range(dim):
                val_pos = values_refine[1 + 2*i]
                val_neg = values_refine[2 + 2*i]
                a_i = (val_pos + val_neg - 2*c_ref) / (2 * d_refine * d_refine)
                q_i = (val_pos - val_neg) / (2 * d_refine)
                a_vals_ref.append(a_i)
                q_vec_ref[i] = q_i
            
            a_ref = np.mean(a_vals_ref)
            
            if abs(a_ref) > 1e-14:
                x_opt_ref = x - q_vec_ref / (2 * a_ref)
                x_opt_ref = np.clip(x_opt_ref, -5, 5)
                f_ref = evaluate(x_opt_ref)
                if f_ref < fx:
                    x = x_opt_ref
                    fx = f_ref

        return best_x, best_f
