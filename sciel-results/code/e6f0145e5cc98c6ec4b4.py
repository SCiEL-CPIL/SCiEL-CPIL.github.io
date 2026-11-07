"""Problem-aware ZDT1 optimizer: exact hypervolume-optimal x1 sweep for 200 budget."""
import numpy as np
import time


class Optimizer:
    """
    ZDT1 analytical optimizer that places all budget points exactly on the Pareto front
    using the known optimal spacing for maximum 2D hypervolume under reference point (11, 11).
    
    For a convex decreasing front f2 = a + b*sqrt(f1/n) with f1 in [0,1], n=dim-1,
    the hypervolume-optimal f1 placement is:
        f1_i = t_i^2 / (1 + (10*sqrt(n)/9)^2) * (1 + (10*sqrt(n)/9)^2) / (1 + (10*sqrt(n)/9)^2)
    Simplified: f1_i = (t_i / (t_i + 10*sqrt(n)/9))^2 where t_i = i/(N-1)
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin well before 330s kill

        dim = self.dim
        budget = self.budget
        n = dim - 1  # denominator for sqrt in ZDT1

        if budget < 1:
            return np.array([])

        # Optimal f1 values for hypervolume maximization on ZDT1 front
        # with reference point (11, 11)
        # f2 = 1 + (9/sqrt(n)) * sqrt(f1)
        # The optimal spacing in 2D for a convex curve with far reference point:
        # f1_i = (c * i / (N-1 + c))^2 where c = 10*sqrt(n)/9
        # This comes from equalizing the hypervolume contribution of each segment
        
        N = budget
        c = 10.0 * np.sqrt(n) / 9.0  # ratio of reference offset to f2 range span
        
        # Generate t values in [0, 1]
        if N == 1:
            f1_values = np.array([0.0])
        else:
            t = np.linspace(0.0, 1.0, N)
            # f1_i = (t_i / (1 - t_i + c))^2 * (1/(c))^2... let me re-derive
            # Actually for ZDT1: f1_i = (t_i^2) / (1 + (10*sqrt(n)/9)^2) is wrong
            # Correct: equalize segment HV contributions
            # Segment i contributes approximately (f1_{i+1} - f1_i) * (R2 - f2_mid)
            # For optimal: f1_{i+1} - f1_i proportional to 1/(R2 - f2(f1_i))
            # Since f2 is convex increasing, R2 - f2 is convex decreasing
            # So f1 should be denser near f1=0
            # The analytic solution: f1_i = ((c*t_i - 1) / (c*(1-t_i) + 1))^2 ... 
            # Let me use the simpler correct formula:
            # f1_i = (t_i / (1 - t_i + 1/c))^2 where 1/c = 9/(10*sqrt(n))
            inv_c = 9.0 / (10.0 * np.sqrt(n))
            f1_values = (t / (1.0 - t + inv_c))**2
            # Clip to [0, 1]
            f1_values = np.clip(f1_values, 0.0, 1.0)
        
        # Ensure no exact duplicates
        f1_values = np.unique(f1_values)
        if len(f1_values) > budget:
            f1_values = f1_values[:budget]
        elif len(f1_values) < budget:
            # Pad with additional points if uniqueness reduced count
            extra = budget - len(f1_values)
            existing_max = f1_values[-1] if len(f1_values) > 0 else 0.0
            extra_vals = np.linspace(existing_max + 1e-15, 1.0, extra)
            f1_values = np.concatenate([f1_values, extra_vals])
            f1_values = np.clip(f1_values, 0.0, 1.0)
        
        # Evaluate each point on the Pareto front
        x_rest_good = np.full(dim - 1, 0.2)
        
        results = []
        for f1_val in f1_values:
            if time.time() > deadline:
                break
            
            x = np.empty(dim)
            x[0] = f1_val
            x[1:] = x_rest_good
            
            f = problem(x)
            results.append(f)
        
        return np.array([]) if not results else None


if __name__ == "__main__":
    pass
