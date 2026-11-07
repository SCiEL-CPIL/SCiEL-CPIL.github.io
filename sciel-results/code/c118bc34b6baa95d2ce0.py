import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.start_time = None
        self.time_limit = 290.0  # 300s limit, keep 10s safety margin

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        
        # Initial point
        # Random start within the box
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = self._evaluate(func, x)
        
        if self._time_exceeded():
            return x, fx

        # Directions: Initially the identity matrix
        directions = np.eye(self.dim)
        
        # Powell's Method
        while self.calls_made < self.budget and not self._time_exceeded():
            x_old = x.copy()
            f_old = fx
            
            # 1. Cyclic line search along each direction
            # We track the direction with the largest improvement to potentially update it
            max_improvement = 0.0
            max_improvement_idx = 0
            
            for i in range(self.dim):
                d = directions[i]
                # Perform 1D minimization along d
                x_new, f_new = self._line_search(func, x, d)
                x = x_new
                fx = f_new
                
                improvement = f_old - fx # Note: f_old is fixed at start of cycle? 
                # Actually, standard Powell compares fx at end of cycle to start.
                # For direction update selection, we look at the improvement gained in that specific step.
                # Let's store the improvement for this step relative to the state *before* this step.
                # But we updated x and fx. 
                # Let's re-calculate improvement for this step:
                # We need f_before_step.
                # Let's refactor to track f_before_step inside the loop.
                
            # Refactored Loop for proper tracking
            x_cycle_start = x_old
            f_cycle_start = f_old
            
            best_improvement = 0
            best_idx = 0
            current_x = x_cycle_start
            current_f = f_cycle_start
            
            for i in range(self.dim):
                if self._time_exceeded() or self.calls_made >= self.budget:
                    break
                d = directions[i]
                f_before_step = current_f
                current_x, current_f = self._line_search(func, current_x, d)
                step_improvement = f_before_step - current_f
                if step_improvement > best_improvement:
                    best_improvement = step_improvement
                    best_idx = i
            
            x = current_x
            fx = current_f
            
            # 2. Check for convergence or direction update
            # If the cycle improved significantly, update the direction with the largest improvement
            # The new direction is the vector from the start of the cycle to the end of the cycle
            if best_improvement > 1e-8: 
                # Calculate new direction
                new_direction = x - x_cycle_start
                norm = np.linalg.norm(new_direction)
                if norm > 1e-8:
                    # Replace the direction that had the best improvement
                    # Standard Powell replaces the direction with the largest drop.
                    # However, we must ensure the new direction is linearly independent.
                    # For Sphere, this is usually fine.
                    directions[best_idx] = new_direction / norm # Normalize? Powell usually doesn't normalize, but it helps with step size consistency.
                    # Actually, Powell's method typically does NOT normalize the new direction.
                    # The line search will handle the scale.
                    directions[best_idx] = new_direction

        return x, fx

    def _evaluate(self, func, x):
        if self.calls_made >= self.budget:
            return 1e10 # Return infinity to prevent further calls
        # Clip to box
        x_clipped = np.clip(x, -5, 5)
        self.calls_made += 1
        try:
            return float(func(x_clipped))
        except Exception:
            return 1e10

    def _time_exceeded(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.time_limit

    def _line_search(self, func, x, d):
        """
        Perform 1D minimization of f(x + t*d).
        Uses a combination of bracketing and parabolic interpolation/golden section.
        """
        if self._time_exceeded() or self.calls_made >= self.budget:
            return x, self._evaluate(func, x) # dummy eval to keep state consistent? No, just return current.
        
        # Initial step size
        # Heuristic: start with small step, expand if improving
        t = 0.1
        f0 = self._evaluate(func, x)
        
        # Try positive direction first
        x1 = x + t * d
        f1 = self._evaluate(func, x1)
        
        if f1 < f0:
            # Improving, try to find the minimum
            # Expand t until f increases
            t_prev = 0.0
            f_prev = f0
            t_curr = t
            f_curr = f1
            
            # Limit expansions to avoid wasting budget
            for _ in range(5):
                if self._time_exceeded() or self.calls_made >= self.budget:
                    break
                t_next = t_curr * 2.0
                x_next = x + t_next * d
                f_next = self._evaluate(func, x_next)
                
                if f_next < f_curr:
                    # Still improving, shift window
                    t_prev, f_prev = t_curr, f_curr
                    t_curr, f_curr = t_next, f_next
                else:
                    # Found a bracket [t_prev, t_next]
                    break
            
            # Now we have a bracket [t_prev, t_curr, t_next] or just [t_prev, t_curr] if we hit limit
            # Use Golden Section or Parabolic Fit on the bracket
            # Points: (t_prev, f_prev), (t_curr, f_curr)
            # If we have a third point, use parabolic.
            # Let's use a simple Golden Section Search on [t_prev, t_next]
            # We already have f_prev and f_curr.
            # Golden section requires evaluating inside.
            
            # To save budget, if we have 3 points, fit parabola.
            # We have (t_prev, f_prev) and (t_curr, f_curr).
            # We need a third point. We have f0 at t=0 if t_prev=0.
            # Let's just do 2-3 Golden Section steps.
            
            a = t_prev
            b = t_next
            # We know f at a (f_prev) and b is not evaluated yet? 
            # In the loop above, if f_next >= f_curr, we broke. So we have f_next.
            # So we have 3 points: (a, f_prev), (c, f_curr) where c is between a and b? No.
            # The loop:
            # 1. t=0.1, f1 < f0. 
            #    t_prev=0, f_prev=f0. t_curr=0.1, f_curr=f1.
            # 2. t_next=0.2. If f_next < f_curr:
            #    t_prev=0.1, f_prev=f1. t_curr=0.2, f_curr=f_next.
            #    ...
            #    If f_next >= f_curr: Break.
            # So we have points at t_prev, t_curr, t_next.
            # f(t_prev) > f(t_curr) < f(t_next).
            # This is a convex bracket.
            
            # Parabolic Fit
            # Fit y = At^2 + Bt + C to (t_prev, f_prev), (t_curr, f_curr), (t_next, f_next)
            # Vertex at t = -B / (2A)
            
            tp, fp = t_prev, f_prev
            tc, fc = t_curr, f_curr
            tn, fn = t_next, f_next
            
            # Check if points are distinct
            if tp == tc or tc == tn or tp == tn:
                 # Fallback to midpoint
                 t_opt = (tp + tn) / 2
            else:
                # Lagrange interpolation for derivative or direct formula
                # A = (fp/(tp*(tp-tn)) + fc/(tc*(tc-tp)) + fn/(tn*(tn-tc)))
                # B = - (fp*(tn+tp)/(tp*(tp-tn)) + fc*(tp+tc)/(tc*(tc-tp)) + fn*(tc+tn)/(tn*(tn-tc)))
                # This is prone to numerical issues.
                
                # Simpler: Golden Section on [tp, tn]
                # We have 2 evaluations already.
                # Let's just do 2 more evaluations for Golden Section.
                # Total 4-5 evals per line search.
                
                invphi = (np.sqrt(5) - 1) / 2
                invphi2 = invphi**2
                
                # We want to minimize on [tp, tn]
                # We have fc at tc.
                # Where is tc relative to the golden points?
                # It's messy to reuse. Let's just do a fresh Golden Section on [tp, tn]
                # but we can use the existing points if they align.
                # For simplicity and robustness, let's just do 3 more evaluations (2 golden + 1 check)
                # or just use the midpoint if budget is tight.
                
                # Given the budget of 500 and ~25 cycles * 5 dims = 125 line searches.
                # 500 / 125 = 4 evals per line search.
                # We used 1 (f1) + expansions.
                # Let's limit expansions to 2.
                
                # Let's just use the vertex of the parabola if it's within [tp, tn]
                # Formula for vertex of 3 points:
                # t_opt = tc - 0.5 * ((tp-tn)*(fc-fn) - (tp-tc)*(fc-fp)) / ((tp-tn)*(fp-fc) - (tp-tc)*(fn-fc))
                # Wait, standard formula:
                # t_opt = tc - 0.5 * ((tp-tn)*(fc-fn) - (tp-tc)*(fc-fp)) / ((tp-tn)*(fp-fc) - (tp-tc)*(fn-fc)) is wrong.
                
                # Let's use:
                # A = (fp - fc)/(tp - tc) - (fn - fc)/(tn - tc)
                # B = (fn - fc)/(tn - tc)
                # t_opt = -B / (2A) ? No.
                
                # Let's stick to Golden Section. It's robust.
                # We have f(tp), f(tn).
                # We need 2 interior points.
                # This costs 2 evals.
                # Total cost: 1 (initial) + k (expansions) + 2 (golden) ~ 4-6.
                # This is too high for 500 budget if we do 50 cycles.
                
                # Alternative: Use the parabolic fit.
                # Let's compute the parabolic vertex.
                # If it's within [tp, tn], use it. Else, use midpoint.
                
                denom = (tp - tn) * (fp - fc) - (tp - tc) * (fn - fc)
                if abs(denom) > 1e-12:
                    num = 0.5 * ((tp - tn) * (fc - fn) - (tp - tc) * (fc - fp))
                    t_parab = tc + num / denom
                    # Check if within bracket
                    if tp <= t_parab <= tn:
                        t_opt = t_parab
                    else:
                        t_opt = (tp + tn) / 2
                else:
                    t_opt = (tp + tn) / 2
            
            x_opt = x + t_opt * d
            f_opt = self._evaluate(func, x_opt)
            
            # Verify improvement
            if f_opt < f_curr:
                return x_opt, f_opt
            else:
                return x_curr, f_curr # x_curr is x + t_curr*d

        else:
            # Try negative direction
            x1_neg = x - t * d
            f1_neg = self._evaluate(func, x1_neg)
            
            if f1_neg < f0:
                # Similar logic for negative direction
                # For brevity, I'll mirror the logic.
                # In a real implementation, I'd factor this out.
                t_prev = 0.0
                f_prev = f0
                t_curr = -t
                f_curr = f1_neg
                t_next = -t * 2.0
                x_next = x + t_next * d
                f_next = self._evaluate(func, x_next)
                
                # If f_next < f_curr, expand once more?
                if f_next < f_curr:
                    t_prev, f_prev = t_curr, f_curr
                    t_curr, f_curr = t_next, f_next
                    t_next = t_next * 2.0
                    x_next = x + t_next * d
                    f_next = self._evaluate(func, x_next)
                
                # Bracket is [t_next, t_curr, t_prev] (note negative values)
                # Sort them: t_min = min(t_next, t_prev), t_max = max(t_next, t_prev)
                # We have 3 points.
                tp, fp = t_prev, f_prev
                tc, fc = t_curr, f_curr
                tn, fn = t_next, f_next
                
                # Ensure tp < tc < tn for the formula?
                # The formula works regardless of order as long as we use the right points.
                # Let's just use the midpoint if unsure, or parabolic fit.
                denom = (tp - tn) * (fp - fc) - (tp - tc) * (fn - fc)
                if abs(denom) > 1e-12:
                    num = 0.5 * ((tp - tn) * (fc - fn) - (tp - tc) * (fc - fp))
                    t_parab = tc + num / denom
                    if min(tp, tn) <= t_parab <= max(tp, tn):
                        t_opt = t_parab
                    else:
                        t_opt = (tp + tn) / 2
                else:
                    t_opt = (tp + tn) / 2
                
                x_opt = x + t_opt * d
                f_opt = self._evaluate(func, x_opt)
                
                if f_opt < f_curr:
                    return x_opt, f_opt
                else:
                    return x + t_curr * d, f_curr
            
            # No improvement in either direction
            return x, f0
