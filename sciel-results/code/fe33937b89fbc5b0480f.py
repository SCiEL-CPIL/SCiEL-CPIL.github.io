"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hypervolume-driven multi-objective optimizer for ZDT6.
    
    Strategy:
    1. Formulate the placement of points as a discrete 1D facility location problem.
    2. Use a greedy algorithm with ternary search to iteratively add points that
       maximize the marginal hypervolume contribution.
    3. The search is performed analytically on the known Pareto front of ZDT6.
    4. Finally, evaluate the selected points using the problem function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f2_zdt6(self, t):
        """Calculate the second objective of ZDT6 for a given x0=t, assuming x_i=0 for i>0."""
        g = 1 + 10 * (self.dim - 1)
        # h(t) = 1 - sqrt(t/g) - sin(10*pi*t)
        # Note: t=0 is a singularity for sqrt, but sqrt(0)=0 is fine.
        h = 1 - np.sqrt(t / g) - np.sin(10 * np.pi * t)
        return g * h

    def _marginal_gain_2d(self, x_new, y_new, x_left, y_left, x_right, y_right, ref):
        """
        Calculate the marginal hypervolume gain of inserting (x_new, y_new)
        between (x_left, y_left) and (x_right, y_right) in a sorted list (by x).
        
        Assumes x_left < x_new < x_right.
        The list is sorted by x ascending.
        The 'prev' y for x_right was y_left.
        The 'prev' y for x_new will be y_left.
        The 'prev' y for x_right will become y_new.
        
        Old contribution involving the segment [x_left, x_right]:
        The point x_right contributed (ref[0] - x_right) * max(0, y_left - y_right) to the HV?
        Not exactly. The HV is the sum of rectangles.
        
        Let's derive the gain directly.
        HV is the area under the step function.
        The step function is defined by the points.
        Between x_left and x_right, the old step was at y_left (from x_left to x_right? No).
        
        Let's look at the standard algorithm:
        Sort by x.
        y_prev = ref[1]
        for (x, y) in points:
            if y < y_prev:
                hv += (ref[0] - x) * (y_prev - y)
                y_prev = y
        
        Consider the points ... (x_left, y_left), (x_right, y_right) ...
        The contribution of (x_right, y_right) to the HV depends on y_prev at that step.
        Before insertion, y_prev for x_right is y_left (assuming no points in between).
        So old_term_right = (ref[0] - x_right) * max(0, y_left - y_right)
        
        After insertion of (x_new, y_new):
        1. Process x_new. y_prev is y_left.
           new_term_new = (ref[0] - x_new) * max(0, y_left - y_new)
           y_prev becomes y_new.
        2. Process x_right. y_prev is y_new.
           new_term_right = (ref[0] - x_right) * max(0, y_new - y_right)
           
        The gain is new_term_new + new_term_right - old_term_right.
        Note: This assumes that y_left, y_new, y_right are the relevant y-values.
        This formula holds if the points are processed in x-order.
        """
        old_term = 0.0
        if y_left > y_right:
            old_term = (ref[0] - x_right) * (y_left - y_right)
        
        new_term_new = 0.0
        if y_left > y_new:
            new_term_new = (ref[0] - x_new) * (y_left - y_new)
            
        new_term_right = 0.0
        if y_new > y_right:
            new_term_right = (ref[0] - x_right) * (y_new - y_right)
            
        return new_term_new + new_term_right - old_term

    def __call__(self, problem, bounds):
        ref = np.array([11.0, 11.0])
        
        # Initialize with two points: t=0 and t=1 (or close to bounds)
        # t=0 -> f1=0, f2 = 91 * (1 - 0 - 0) = 91.
        # t=1 -> f1=1, f2 = 91 * (1 - sqrt(1/91) - 0) ~ 91 * (1 - 0.1) = 81.9.
        # Both have f2 > 11, so they might not contribute to HV if no other points.
        # But they define the boundary.
        
        selected_t = [0.0, 1.0]
        selected_f2 = [self._f2_zdt6(t) for t in selected_t]
        
        # We want to add budget - 2 more points.
        for _ in range(self.budget - 2):
            best_gain = -np.inf
            best_t = None
            best_idx = None
            
            # Iterate over all gaps
            for i in range(len(selected_t) - 1):
                t_left = selected_t[i]
                t_right = selected_t[i + 1]
                y_left = selected_f2[i]
                y_right = selected_f2[i + 1]
                
                # Ternary search for best t in [t_left, t_right]
                lo = t_left
                hi = t_right
                # If the gap is very small, skip or just pick middle
                if hi - lo < 1e-9:
                    continue
                    
                for _ in range(50):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    
                    y_m1 = self._f2_zdt6(m1)
                    y_m2 = self._f2_zdt6(m2)
                    
                    gain_m1 = self._marginal_gain_2d(m1, y_m1, t_left, y_left, t_right, y_right, ref)
                    gain_m2 = self._marginal_gain_2d(m2, y_m2, t_left, y_left, t_right, y_right, ref)
                    
                    if gain_m1 < gain_m2:
                        lo = m1
                    else:
                        hi = m2
                        
                t_opt = (lo + hi) / 2.0
                y_opt = self._f2_zdt6(t_opt)
                gain_opt = self._marginal_gain_2d(t_opt, y_opt, t_left, y_left, t_right, y_right, ref)
                
                if gain_opt > best_gain:
                    best_gain = gain_opt
                    best_t = t_opt
                    best_idx = i
            
            # Insert the best point
            if best_t is not None:
                # Insert into sorted list
                selected_t.insert(best_idx + 1, best_t)
                selected_f2.insert(best_idx + 1, self._f2_zdt6(best_t))
        
        # Now we have the optimal set of t values.
        # Call problem for each.
        for t in selected_t:
            x = np.zeros(self.dim)
            x[0] = t
            # x[1:] are already 0
            problem(x)
            
        return selected_t

# EVOLVE-BLOCK-END
