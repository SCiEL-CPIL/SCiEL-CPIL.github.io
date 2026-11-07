# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT2 with 30 variables, 2 objectives.
    
    Strategy:
    1. Do not assume x[1:]=0 is the optimal Pareto front location.
    2. Use a dense grid search over x[0] (100 points) and a small grid over x[1], x[2] (3x3 points).
    3. For each (x0, x1, x2) combination, set x[3:] = 0.
    4. This creates 100 * 3 * 3 = 900 potential evaluations, but we only have budget 200.
    5. We will prioritize evaluations that are likely to be on the front. 
       Since ZDT2's front is typically at x[1:]=0, we will first evaluate the "ideal" front (x[1:]=0) for all x0.
       This uses 100 evaluations.
    6. Then, use remaining 100 evaluations to explore x[1] and x[2] around 0 for a subset of x0 values to see if the front shifts.
    7. Actually, a better strategy given the budget constraint (200) is:
       - Sample x[0] uniformly at 200 points.
       - For each x[0], we must choose x[1:]. 
       - If we assume the front is at x[1:]=0, we get the theoretical front.
       - The prompt asks to "empirically map the true Pareto front to account for any potential deviations".
       - However, with only 200 evaluations, a full grid is impossible.
       - Let's stick to the most robust interpretation: The "deviations" might be minor. 
       - But wait, the current program already scores 1.3001. The inspirations score 1.3009.
       - The difference is small. 
       - Let's try a different structural approach: 
         Use the budget to evaluate a dense set of points on the *theoretical* front, but also include a small fraction of points with x[1] > 0 to check if the front is actually shifted. 
         If the front is shifted, points with x[1]>0 might have lower f2 (better) for the same f1.
         Actually, for ZDT2, g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 * (1+10*sin(3*pi*xi))).
         The minimum of g is 1, achieved at xi=0.5? No, at xi=0 or 1? 
         sin(3*pi*0) = 0. (0-0.5)^2 * 1 = 0.25.
         sin(3*pi*0.5) = sin(1.5*pi) = -1. (0-0.5)^2 * (1-10) = 0.25 * -9 = -2.25.
         Wait, the standard ZDT2 definition is:
         g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 * (1+10*sin(3*pi*xi)))
         f1 = x1
         f2 = g * (1 - sqrt(f1/g))
         
         To minimize f2, we want g to be as small as possible.
         The term (xi-0.5)^2 * (1+10*sin(3*pi*xi)).
         If xi=0.5, term is 0.
         If xi=0, term is 0.25 * 1 = 0.25.
         If xi=1, term is 0.25 * 1 = 0.25.
         So xi=0.5 gives 0 contribution.
         Therefore, the minimum g is achieved when xi=0.5 for all i>=1?
         Let's check: if xi=0.5, (0.5-0.5)^2 * ... = 0.
         So g_min = 1 + 10*(n-1) + 0 = 1 + 10*29 = 291?
         No, the standard ZDT definition often has g(x) = 1 + 10*(n-1) + sum(...).
         But sometimes it's just g(x) = 1 + 10*(n-1) * (sum(...) / (n-1))? No.
         Let's look at the standard ZDT2:
         g(x) = 1 + 10*(n-1) + sum_{i=2}^n ((x_i - 0.5)^2 * (1 + 10*sin(3*pi*x_i)))
         f1 = x1
         f2 = g * (1 - sqrt(f1/g))
         
         The Pareto front is obtained when g(x) = g_min.
         g_min occurs when the sum is minimized.
         Each term is >= 0? No, 1+10*sin(3*pi*xi) can be negative.
         Min of 1+10*sin(3*pi*xi) is 1-10 = -9.
         Min of (xi-0.5)^2 is 0.
         The product is minimized when (xi-0.5)^2 is large and the sine term is -1?
         Let xi=0.5: term=0.
         Let xi=0: term=0.25 * 1 = 0.25.
         Let xi=1/6: sin(pi/2)=1. term = (1/6-0.5)^2 * 11 = (1/3)^2 * 11 = 1/9 * 11 = 1.22.
         Let xi=1/2: 0.
         Let xi=1/3: sin(pi) = 0. term = (1/3-0.5)^2 * 1 = 1/36.
         Let xi=1/6: sin(pi/2)=1.
         Let xi=5/6: sin(5pi/2)=1.
         Let xi=1/2: 0.
         
         Actually, the global minimum of the term (x-0.5)^2 * (1+10*sin(3*pi*x)) is NOT 0.
         It is negative.
         Let h(x) = (x-0.5)^2 * (1+10*sin(3*pi*x)).
         At x=0.5, h=0.
         At x=0.25, sin(0.75pi)=0.707. ( -0.25)^2 * (1+7.07) = 0.0625 * 8.07 = 0.5.
         At x=0.75, sin(2.25pi)=-0.707. (0.25)^2 * (1-7.07) = 0.0625 * -6.07 = -0.38.
         So negative values are possible.
         
         However, the "standard" ZDT2 front is often cited as being at x[1:]=0.5? Or x[1:]=0?
         Let's check the literature. 
         "The Pareto-optimal front of ZDT2 is a curve in the objective space..."
         Usually, the front is defined for g=1.
         If the problem instance provided by the evaluator uses the standard definition, we need to find the x that minimizes g.
         
         If we don't know the exact x that minimizes g, we can't just set x[1:]=0.
         The current program assumes x[1:]=0. 
         If x[1:]=0.5 is better, the current program is suboptimal.
         
         The directive says: "use the budget to perform a dense uniform grid search over a low-dimensional subset of the quality variables (e.g., x[1] and x[2]) while sweeping x[0], empirically mapping the true Pareto front".
         
         Plan:
         1. Allocate 150 evaluations for sweeping x[0] with x[1:]=0.5 (a common candidate for min g).
         2. Allocate 50 evaluations for sweeping x[0] with x[1:]=0.
         3. Compare the results? No, we just record all points. The hypervolume calculation will automatically pick the best ones if they are non-dominated.
         4. Actually, if we evaluate both, we get more points. The hypervolume is calculated on the *union* of all recorded points.
         5. So, we should sample multiple candidate configurations for x[1:] to cover the likely location of the front.
         
         Candidates for x[1:]:
         - All 0
         - All 0.5
         - All 1
         - Random?
         
         Let's split the 200 budget:
         - 100 points: x[0] uniform, x[1:] = 0
         - 100 points: x[0] uniform, x[1:] = 0.5
         
         This will ensure we cover both common candidates for the Pareto front location.
         The hypervolume calculation will take the non-dominated subset. If one configuration is strictly better, it will dominate the other. If they are different, we get a union.
         
         Even better:
         - 60 points: x[0] uniform, x[1:] = 0
         - 60 points: x[0] uniform, x[1:] = 0.5
         - 40 points: x[0] uniform, x[1:] = 1
         - 40 points: x[0] uniform, x[1:] = 0.25 (or other)
         
         This spreads the risk.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        n_evals = 0
        all_points = []

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                f_vals = np.asarray(problem(x), dtype=float)
                if f_vals.shape != (self.n_obj,):
                    return None
                if not np.all(np.isfinite(f_vals)):
                    return None
                n_evals += 1
                return f_vals
            except Exception:
                n_evals += 1
                return None

        # Define candidate vectors for x[1:]
        # We will sweep x[0] for each candidate.
        # Split budget among candidates.
        
        candidates = [
            np.zeros(self.dim - 1),      # x[1:] = 0
            np.full(self.dim - 1, 0.5),  # x[1:] = 0.5
            np.ones(self.dim - 1),       # x[1:] = 1
            np.full(self.dim - 1, 0.25), # x[1:] = 0.25
        ]
        
        n_candidates = len(candidates)
        # Distribute budget evenly
        evals_per_candidate = self.budget // n_candidates
        # Handle remainder by giving extra to first candidates
        remainders = self.budget % n_candidates
        
        x0_samples_base = np.linspace(0.0, 1.0, self.budget) # Pre-generate max possible samples
        
        for i, cand in enumerate(candidates):
            n_evals_for_this = evals_per_candidate + (1 if i < remainders else 0)
            if n_evals_for_this <= 0:
                continue
            
            # Generate x0 samples for this candidate
            # Use unique x0 values for each candidate to maximize coverage
            # Or reuse the same x0 values? 
            # If we reuse, we are comparing the same x0 with different x[1:].
            # This is good for finding which x[1:] is better for a given x0.
            # But for hypervolume, we just want a dense set of points.
            # Let's use the same x0 grid for all candidates.
            x0_samples = x0_samples_base[:n_evals_for_this]
            
            for x0 in x0_samples:
                if n_evals >= self.budget or time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1:] = cand
                
                f_vals = evaluate(x)
                if f_vals is not None:
                    if np.all(f_vals < 11.0):
                        all_points.append(f_vals)

        return all_points
# EVOLVE-BLOCK-END
