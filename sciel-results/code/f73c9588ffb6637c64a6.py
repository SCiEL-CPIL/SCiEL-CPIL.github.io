"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT6 structure.
    
    Strategy:
    1. Analyze ZDT6:
       f1(x) = 1 + 10*(n-1) + sum_{i=2}^n (xi - 0.5)^2 * (1 + 10*sin(6*pi*xi + pi*f1))
       f2(x) = g(x) * h(g(x)/f1(x))
       g(x) = 1 + 9 * sum_{i=2}^n xi / (n-1)
       h(r) = 1 - sqrt(r/f1) - (r/f1)^2
       
       For ZDT6, the Pareto front is where f1 is minimized (f1=1) and g varies from 1 to 11.
       Actually, f1 depends on x1..xn. But typically for ZDT4/6, the front is achieved when 
       x2..xn are set to minimize g (i.e., xi=0 for i>1) or vary to sweep g.
       
       Wait, for ZDT6:
       f1 = 1 + 10*(n-1) + ...
       If we set x2..xn = 0, then sum=0, so f1 = 1.
       Then g = 1.
       f2 = 1 * h(1/1) = 1 - 1 - 1 = -1? No, h(r) = 1 - sqrt(r/f1) - (r/f1)^2.
       If f1=1, r=g=1, h(1) = 1 - 1 - 1 = -1.
       Wait, the standard ZDT6 definition has h(r) = 1 - sqrt(r) - r^2 for f1=1?
       Let's check the contract: "minimizing all 2 objectives".
       Usually ZDT6 front is f1=1, f2 in [0,11].
       
       Let's look at the reference point [11.0, 11.0].
       If f1=1, max f2 is 11 (when g=11).
       g=11 when sum_{i=2}^n xi = n-1 (i.e., all xi=1).
       
       So the Pareto front is characterized by:
       f1 = 1 (min possible)
       f2 varies from 0 to 11 as g varies from 1 to 11.
       
       To achieve f1=1, we need the summation term in f1 to be 0.
       f1 = 1 + 10*(n-1) + sum_{i=2}^n (xi - 0.5)^2 * (1 + 10*sin(6*pi*xi + pi*f1))
       This is tricky because f1 is inside the sin term.
       However, for ZDT4/6, it is known that setting x2..xn to 0 or 1 might not yield f1=1 exactly due to the sin term.
       BUT, there is a known property: if x_i = 0.5, the term is 0.
       If x_i = 0.5 for all i>1, then sum=0, f1=1.
       Then g = 1 + 9 * sum(0.5) / (n-1) = 1 + 4.5 = 5.5.
       This gives a single point.
       
       To sweep the front, we need to vary g while keeping f1=1.
       This requires solving for x_i such that the f1 penalty is 0 and g takes desired values.
       This is hard analytically.
       
       Alternative Strategy:
       The current program scores 1.0. This implies it already finds the front well enough to match the "dense fixed sample" normalization?
       No, "score 1.0 equals the seed program".
       The seed program is a simple random search.
       
       Let's look at the inspirations. They scored lower (0.5, 0.6).
       This suggests the seed program is actually quite good, or the scoring is tricky.
       
       Let's rethink the ZDT6 structure.
       If we simply sample points where x2..xn are set to values that produce a spread of g, and optimize x1..xn to minimize f1 locally?
       
       A robust approach for ZDT6 with budget 200:
       1. Generate a set of target g-values (e.g., 20 points linearly from 1 to 11).
       2. For each target g, construct a candidate x where x2..xn are set to achieve that g (e.g., x2 = (g-1)*(n-1)/9, x3..xn=0).
       3. Perform a local search (e.g., Nelder-Mead or simple gradient-free steps) on these candidates to minimize f1, while monitoring f2.
       
       However, minimizing f1 is hard because of the sin term.
       
       Let's try a different angle: **Hypervolume Maximization via Direct Sampling**.
       The reference point is [11, 11].
       The front is roughly f1=1, f2=11(1-sqrt(f1/11))? No, that's ZDT1.
       ZDT6 front: f1=1, f2 in [0, 11].
       Actually, for ZDT6, f2 = g * h(g/f1).
       If f1=1, f2 = g * (1 - sqrt(g) - g^2).
       Max g=11. h(11) = 1 - sqrt(11) - 121 < 0.
       Wait, the objectives are to be MINIMIZED.
       If f2 becomes negative, it's "better" (smaller).
       But the reference point is [11, 11].
       Points with f2 < 0 are strictly better than the reference.
       
       Let's check the ZDT6 definition carefully.
       Standard ZDT6:
       f1(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i - 0.5)^2 (1 + 10 sin(3 pi x_i))
       f2(x) = g(x) h(f1(x)/g(x))
       g(x) = 1 + 9 (sum_{i=2}^n x_i)/(n-1)
       h(r) = 1 - sqrt(r) - r^2
       
       If f1=1, g=1, r=1, h(1) = 1-1-1 = -1. f2 = -1.
       If f1=1, g=11, r=1/11, h(1/11) = 1 - sqrt(1/11) - (1/11)^2 approx 1 - 0.3 - 0.008 = 0.69. f2 = 11 * 0.69 = 7.6.
       
       So the front spans f2 from -1 to ~7.6?
       Or does g vary differently?
       
       Regardless of the exact curve, the key is that **f1 is the primary driver of the front's position in the 1st objective**, and **g drives the 2nd**.
       
       The most effective strategy with a small budget (200) is often **Uniform Sampling in the decision space** followed by **Non-Dominated Sorting**, OR **Sampling along the expected Pareto Front**.
       
       Since the seed program (random search) scores 1.0, and other complex attempts scored lower, it is possible that random search is surprisingly effective for ZDT6 because the front is "flat" or the penalty terms are not too restrictive.
       
       However, we can do better by **biasing the random search**.
       Instead of uniform random, we can sample x1 uniformly, and for x2..xn, sample them such that g is spread out.
       
       Let's implement a **Stratified Sampling** approach:
       1. Divide the budget into 200 evaluations.
       2. Create 200 candidate points.
       3. For each candidate, choose a target g value from a uniform distribution in [1, 11].
       4. Set x2..xn to achieve this g (e.g., x2 = (g-1)*(n-1)/9, others 0).
       5. Set x1 to a random value in [0,1].
       6. Evaluate.
       7. Keep the non-dominated set.
       
       This ensures we cover the entire range of g (and thus f2) without wasting evaluations on redundant g values.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use the entire budget for stratified sampling.
        # n_samples = self.budget
        n_samples = self.budget
        
        # Generate stratified g values
        # g = 1 + 9 * sum(x[1:]) / (n-1)
        # To get a specific g, we can set x[1] (which is x2 in 0-indexed) to carry the load.
        # x[1] = (g - 1) * (n - 1) / 9
        # Other x[i] for i > 1 can be 0.5 (center) or random.
        
        # We want to minimize f1.
        # f1 penalty is sum (xi - 0.5)^2 * (...)
        # To minimize penalty, we should keep xi close to 0.5 if possible.
        # But we need to set x[1] to a specific value for g.
        # For other xi, we can try 0.5.
        
        candidates = []
        for i in range(n_samples):
            # Target g uniformly in [1, 11]
            g_target = 1 + 10 * (i / (n_samples - 1))
            
            # Calculate required x[1] (index 1)
            # g = 1 + 9 * x[1] / (n-1)  (assuming x[2..n-1] = 0)
            # x[1] = (g - 1) * (n - 1) / 9
            x1_val = (g_target - 1) * (self.dim - 1) / 9.0
            
            # Clip to bounds
            x1_val = np.clip(x1_val, lo[1], hi[1])
            
            x = np.zeros(self.dim)
            x[0] = np.random.uniform(lo[0], hi[0]) # x1 (index 0) is free for f1 optimization?
            # Actually, f1 depends on x1 too?
            # ZDT6: f1 = 1 + 10(n-1) + sum_{i=2}^n ...
            # The sum is from i=2 to n. In 0-indexed numpy, that's indices 1 to n-1.
            # So x[0] (x1) does NOT appear in the sum!
            # It only appears in the term 1 + 10(n-1)? No, that's constant.
            # Wait, standard ZDT6:
            # f1 = 1 + 10(n-1) + sum_{i=2}^n (x_i - 0.5)^2 (1 + 10 sin(3 pi x_i))
            # Yes, x1 is not in the sum.
            # So x1 can be anything?
            # But f2 depends on g and f1.
            # If x1 is not in f1 or g, it doesn't affect objectives?
            # Let's check the definition again.
            # Usually ZDT1: f1 = x1.
            # ZDT4/6: f1 is the complex sum. x1 is often the "shape" variable.
            # If x1 is not in f1, then f1 is constant for a given x2..xn?
            # No, the sum is over i=2..n.
            # So x1 is irrelevant to f1?
            # If so, we can set x1 to 0.5 to be safe.
            
            x[1] = x1_val
            # For remaining dimensions, set to 0.5 to minimize the (xi-0.5)^2 term
            for j in range(2, self.dim):
                x[j] = 0.5
                
            candidates.append(x)
        
        # Evaluate all candidates
        F = np.array([problem(x) for x in candidates])
        
        # Check time
        if time.time() > deadline - 10:
            return candidates
            
        # Now, we have a set of points.
        # We can try to improve the worst points or refine the non-dominated set.
        # Let's identify non-dominated points.
        
        def is_dominated(f, F):
            # f is dominated if there exists g in F such that g <= f and g != f
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
        dominated = np.array([is_dominated(f, F) for f in F])
        non_dom_idx = np.where(~dominated)[0]
        
        # If we have time, refine the non-dominated points?
        # Or just return the candidates.
        # The contract says "Every returned objective vector is recorded".
        # So we should return the list of x's?
        # "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This implies the evaluator calls problem(x) for each x we return?
        # No, "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded"
        # This phrasing is ambiguous. Does it mean the evaluator records the outputs of problem(x) *during* the call?
        # "More than 200 evaluations... makes the whole result invalid"
        # This implies the evaluator counts calls to `problem`.
        # So the evaluator is likely wrapping `problem` or counting calls.
        # "The return value of __call__ is ignored."
        # So we just need to call `problem` up to 200 times.
        # The evaluator will collect the (f1, f2) pairs from those calls.
        
        # So my strategy of calling problem 200 times in the loop above is correct.
        # I don't need to return anything meaningful.
        
        return candidates
# EVOLVE-BLOCK-END
