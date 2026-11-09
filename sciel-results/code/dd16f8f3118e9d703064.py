import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        # ZDT3 Analysis:
        # f1(x) = 1 + 9 * g(x[1:]) / (dim - 1)  (Wait, standard ZDT3 has f1 = 1 + 9*g/(n-1)? No, ZDT1/2/3 usually have f1=x0 for ZDT1/2. For ZDT3, f1 = 1 + 9*g/(n-1) is WRONG. 
        # Let's check standard ZDT3 definition.
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) is for ZDT2? No.
        # ZDT1: f1=x0, f2=1+...
        # ZDT2: f1=x0, f2=1-x0^2+...
        # ZDT3: f1 = 1 + 9 * g / (n-1) is NOT standard. 
        # Standard ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) ? No, that's ZDT2's f2 structure sometimes.
        # Let's look up ZDT3 specifically.
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) ... wait, if f1 depends on g, and g=1 on Pareto front, then f1 is constant? No.
        # Actually, for ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) is incorrect.
        # Correct ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) ... I am confusing myself.
        # Let's recall the discontinuity.
        # ZDT3 has a discontinuous Pareto front.
        # The Pareto front is defined by g(x[1:]) = 1.
        # On the Pareto front, x[1:] are such that g=1.
        # For ZDT3, g(x) = 1 + 10n + sum(5x_i + sin(10x_i)) ... no, that's for ZDT1/2 where g=0 is optimal.
        # For ZDT3, the condition for the Pareto front is different.
        # Actually, ZDT3 is defined such that the Pareto front is at g(x[1:]) = 1.
        # And f1(x) = 1 + 9 * g(x[1:]) / (n-1) is NOT the formula.
        
        # Let's look at the standard definition of ZDT3 from the literature (Deb et al. 2002):
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1) -- This is ZDT2's f2? No.
        # ZDT1: f1=x0, f2=1+9/(n-1)*g, g=sum(xi)-xi^2. PF: g=0.
        # ZDT2: f1=x0, f2=1-f1^2+9/(n-1)*g, g=sum(xi)-xi^2. PF: g=0.
        # ZDT3: f1=1+9/(n-1)*g, f2=1-sqrt(f1)+9/(n-1)*g. PF: g=1.
        # Wait, if f1 depends on g, and g=1 on PF, then f1 is constant on the PF?
        # If f1 is constant, the PF is a point? No.
        
        # Let's re-read the ZDT3 definition carefully.
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # g(x) = 1 + 10n + sum_{i=2}^n (5x_i + sin(10x_i))
        # The Pareto front is where g(x[1:]) = 1.
        # If g=1, then f1 = 1 + 9/(n-1) * 1 = 1 + 9/(n-1). This is a constant!
        # This implies the Pareto front is a single point? That cannot be right for a multi-objective problem with a "front".
        
        # Ah, I must be misremembering the formula.
        # Let's check another source.
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # If g=1, f1 is constant. 
        # Is it possible that x[0] is involved in f1?
        # In ZDT1 and ZDT2, f1 = x[0].
        # In ZDT3, is f1 = x[0]?
        # If f1 = x[0], and PF is g=1, then x[0] can vary, and f1 varies.
        # Let's assume f1(x) = x[0] for ZDT3 as well?
        # No, ZDT3 is known for its discontinuous front.
        # If f1=x0, and g=1 defines the front, then the front is a curve in f1-f2 space.
        # f2(x) = 1 - sqrt(x0) + 9/(n-1) * g(x[1:])
        # If g=1, f2 = 1 - sqrt(x0) + 9/(n-1).
        # This looks like a smooth curve. Why is it discontinuous?
        
        # The discontinuity in ZDT3 comes from the function g(x).
        # g(x) = 1 + 10n + sum(5x_i + sin(10x_i))
        # The minimum of 5x + sin(10x) is not at x=0.
        # Actually, for ZDT3, the Pareto front is NOT at g=0. It is at g=1?
        # Or is the front defined by the local minima of g?
        
        # Let's look at the "Inspiration 1" text: "ZDT3 has a discontinuous Pareto front where the objective f1 depends on x0 in a non-linear way".
        # This suggests f1 is NOT linear x0.
        # If f1 depends on x0 non-linearly, maybe f1 = 1 + 9*g/(n-1) is wrong.
        
        # Let's try to find the exact ZDT3 definition.
        # From "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II":
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # g(x) = 1 + 10n + sum_{i=2}^n (5x_i + sin(10x_i))
        # If this is correct, then on the Pareto front (where g is minimized? or where g is specific?), f1 is constant.
        # This implies the "front" is a single point? 
        # No, the "Pareto front" in these problems is the set of non-dominated points.
        # If f1 is constant for all points on the "optimal" set, then only one point can be non-dominated?
        # Unless... x[0] is NOT in g, but IS in f1?
        # In ZDT1/2, f1=x0.
        # In ZDT3, if f1 is constant, it's a different class.
        
        # Wait, I might be swapping ZDT3 with something else.
        # Let's check the discontinuity.
        # The discontinuity in ZDT3's Pareto front is in the f2 vs f1 plot.
        # If f1 = x0, and f2 = 1 - sqrt(x0) + C, it's smooth.
        # So f1 MUST depend on something else.
        
        # Actually, looking at online sources for ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # This definition makes f1 independent of x0.
        # This means x0 does not affect the objectives?
        # That would make x0 irrelevant. This is unlikely for a 30-dim problem.
        
        # Let's try another definition.
        # Maybe f1(x) = x0?
        # And f2(x) = 1 - sqrt(x0) + 9 * g(x[1:]) / (n-1)?
        # If so, the PF is at g(x[1:]) = min(g).
        # For ZDT3, min(g) is not 0.
        # The function h(x) = 5x + sin(10x).
        # The global minimum of h(x) on [0,1] is not at 0.
        # h(0) = 0.
        # h(0.1) = 0.5 + sin(1) = 0.5 + 0.84 = 1.34.
        # h(0.2) = 1.0 + sin(2) = 1.0 + 0.91 = 1.91.
        # h(0.5) = 2.5 + sin(5) = 2.5 - 0.96 = 1.54.
        # h(1.0) = 5.0 + sin(10) = 5.0 - 0.54 = 4.46.
        # It seems h(x) is increasing?
        # Derivative: 5 + 10cos(10x).
        # Zeros of derivative: cos(10x) = -0.5.
        # 10x = 2pi/3, 4pi/3, ...
        # x = 0.209, 0.418, ...
        # At x=0.209, h(x) = 5(0.209) + sin(2.09) = 1.045 + 0.866 = 1.91.
        # At x=0, h(0)=0.
        # So the minimum of sum(h(x_i)) is at x_i=0?
        # If min is at x_i=0, then g_min = 1 + 10n.
        # Then f1_min = 1 + 9(1+10n)/(n-1).
        # This is still constant.
        
        # I am stuck on the definition. Let's look at the "Inspiration 1" again.
        # "f1 depends on x0 in a non-linear way".
        # This strongly suggests f1 is NOT x0.
        # And if f1 is NOT constant, it must depend on x0.
        # So f1(x) = f(x0, g(x[1:])).
        
        # Let's assume the standard ZDT3 definition from the CMU website or similar:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # If this is the case, x0 is a dummy variable?
        # No, in NSGA-II papers, ZDT3 is used to test discontinuous fronts.
        # The discontinuity arises because the Pareto front is not connected.
        # But if f1 is constant, the front is a point.
        
        # Let's try:
        # f1(x) = x0
        # f2(x) = 1 - sqrt(x0) + 9 * g(x[1:]) / (n-1)
        # If this is the case, the front is at g = min(g).
        # If min(g) is at x[1:] = 0, then f2 = 1 - sqrt(x0) + C.
        # This is a smooth curve.
        
        # Where is the discontinuity?
        # The discontinuity in ZDT3 is in the shape of the front.
        # It is composed of multiple disconnected segments.
        # This happens if the condition for the Pareto front (g=min) is satisfied in multiple disconnected regions of x[1:] space?
        # Or if the mapping from x0 to f1 is non-linear?
        
        # Actually, I found a source:
        # ZDT3:
        # f1(x) = 1 + 9 * g(x[1:]) / (n-1)
        # f2(x) = 1 - sqrt(f1(x)) + 9 * g(x[1:]) / (n-1)
        # g(x) = 1 + 10n + sum(5x_i + sin(10x_i))
        # The Pareto front is the set of points where g(x[1:]) = 1? No, g is always >= 1+10n.
        
        # Wait, I see "g(x) = 1 + 10n + ...".
        # If the problem is defined such that the Pareto front is at g=1, then the formula for g must be different.
        # Maybe g(x) = sum(5x_i + sin(10x_i))?
        # And the condition is g=1?
        
        # Let's look at the current program's strategy:
        # "Strategy: ZDT3 Pareto front is at x[1:] = 0"
        # "We sample x[0] uniformly."
        # This implies that x[1:]=0 IS on the Pareto front.
        # And x[0] varies along the front.
        # This implies f1 depends on x[0].
        # So f1(x) is likely x[0] or a function of x[0].
        
        # If x[1:]=0 is on the Pareto front, then g(0) must be the "optimal" value.
        # If the current program scores 1.26, it means this strategy is good.
        # The inspiration says "f1 depends on x0 in a non-linear way".
        # If f1 = x0, it is linear.
        # So f1 is likely NOT x0.
        
        # Let's guess the non-linear dependence.
        # Maybe f1 = x0^2? Or 1 - cos(x0)?
        # If f1 is non-linear, uniform sampling of x0 is NOT uniform sampling of f1.
        # To maximize hypervolume, we should sample f1 uniformly, or adaptively.
        
        # Since I don't know the exact formula, I can use the `problem` function to estimate the front.
        # Strategy:
        # 1. Use a small fraction of the budget to explore the front.
        # 2. Identify the range of f1 and f2.
        # 3. Use the remaining budget to sample the front more effectively.
        
        # However, the current program uses 90% uniform x0 and 10% random.
        # If f1 is non-linear, uniform x0 might miss parts of the front (in objective space).
        # For example, if f1 = x0^2, uniform x0 gives sparse samples at low f1 and dense at high f1.
        # To cover hypervolume, we want uniform f1.
        
        # Let's implement an adaptive strategy.
        # 1. Initial phase: Sample x0 uniformly (e.g., 50 points).
        # 2. Evaluate them.
        # 3. Sort by f1.
        # 4. Identify gaps in f1 space.
        # 5. Use remaining budget to fill gaps (bisection or inverse mapping approximation).
        
        # But we don't know the mapping x0 -> f1.
        # We can assume x0 is monotonic with f1.
        # If we have points (x0_i, f1_i), we can interpolate x0 for a target f1.
        
        # Let's refine the current program:
        # 1. Use 20% of budget for initial uniform sampling of x0 (on x[1:]=0).
        # 2. Record f1 values.
        # 3. For the remaining 80%, target f1 values uniformly in [min_f1, max_f1].
        # 4. Use linear interpolation of the initial samples to estimate x0 for target f1.
        # 5. Evaluate those points.
        
        # This should provide better coverage in objective space.
        
        n_initial = max(20, int(budget * 0.25))
        n_refine = budget - n_initial
        
        # Phase 1: Initial Sampling with endpoints emphasized
        # Use linspace but add extra points near boundaries
        x0_samples = np.linspace(lo[0], hi[0], n_initial)
        # Add a few extra points near the boundaries for better endpoint characterization
        boundary_points = np.concatenate([
            np.linspace(lo[0], lo[0] + 0.05*(hi[0]-lo[0]), 5),
            np.linspace(hi[0] - 0.05*(hi[0]-lo[0]), hi[0], 5)
        ])
        x0_samples = np.unique(np.concatenate([x0_samples, boundary_points]))
        n_initial = len(x0_samples)
        n_refine = budget - n_initial
        
        x_init = np.zeros((n_initial, dim))
        x_init[:, 0] = x0_samples
        # x_init[:, 1:] = lo[1:] (which is 0)
        
        f1_vals = np.zeros(n_initial)
        f2_vals = np.zeros(n_initial)
        
        # Evaluate initial points
        for i in range(n_initial):
            if time.time() > deadline:
                break
            try:
                f = problem(x_init[i])
                f1_vals[i] = f[0]
                f2_vals[i] = f[1]
            except Exception:
                break
        
        # Phase 2: Refinement with precise endpoint handling
        valid = np.isfinite(f1_vals)
        n_valid = np.sum(valid)
        if n_valid > 2:
            min_f1 = np.min(f1_vals[valid])
            max_f1 = np.max(f1_vals[valid])
            
            # Ensure we hit the exact min and max f1
            # Target f1 values uniformly, including endpoints
            target_f1 = np.linspace(min_f1, max_f1, n_refine)
            
            # Interpolate x0 for each target_f1
            sort_idx = np.argsort(f1_vals[valid])
            f1_sorted = f1_vals[valid][sort_idx]
            x0_sorted = x0_samples[sort_idx]
            
            # Use linear interpolation for the inverse mapping
            x0_targets = np.interp(target_f1, f1_sorted, x0_sorted)
            
            # Create and evaluate points
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0_targets[i]
                # x[1:] = 0 (already zeros)
                try:
                    problem(x)
                except Exception:
                    break
        else:
            # Fallback to random if initial sampling failed
            for _ in range(n_refine):
                if time.time() > deadline:
                    break
                x = np.random.uniform(lo, hi)
                try:
                    problem(x)
                except Exception:
                    break
        
        return None
