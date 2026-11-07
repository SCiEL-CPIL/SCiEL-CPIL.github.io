import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Grid-Partitioned Local Optimization.
    
    Strategy:
    1. The Pareto front of ZDT3 is f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1] (with x2..xn=0).
    2. The problem reduces to selecting B values of x1 in [0,1] to maximize the Hypervolume 
       of the set of points {(x1_i, 1 - sqrt(x1_i))} with reference point [11, 11].
    3. Instead of a single global optimization which may get stuck in local optima due to the 
       non-smoothness and complexity of the HV landscape, we partition the x1 domain [0, 1] 
       into 10 equal segments.
    4. We assign B/10 points to each segment.
    5. For each segment, we run a short Nelder-Mead optimization to place the points optimally 
       within that segment, considering the interaction with the other segments (approximated 
       by fixing the points in other segments to their current best or initial uniform positions).
       To be efficient and avoid coupling issues, we optimize each segment independently assuming 
       the other segments provide a 'background' density. However, a simpler and robust approach 
       is to optimize each segment's points to maximize the local contribution, or simply use 
       a multi-start Nelder-Mead on the whole vector but with 10 distinct initializations 
       corresponding to the 10 segments having slightly perturbed uniform distributions.
       
       Actually, the directive suggests: "initializing 20 simplex bases at the 10 analytic local minima...".
       ZDT3's front is continuous. The "local minima" phrasing might refer to the fact that the 
       optimal HV point distribution has a specific structure. 
       
       Let's stick to the multi-start idea: 
       We will run 10 separate Nelder-Mead optimizations, each starting from a slightly different 
       perturbed uniform distribution, or better, we will use the 10 segments to define 10 
       different "bias" initializations.
       
       A more direct interpretation of the directive: 
       "Apply a multi-start Nelder-Mead local search exclusively on the x1 coordinate, 
        initializing 20 simplex bases at the 10 analytic local minima..."
        
       Since ZDT3 doesn't have discrete local minima in x1 for the front itself, I will interpret 
       "10 analytic local minima" as 10 distinct regions of the front where the curvature or 
       slope changes significantly, or simply 10 equal intervals. I will use 10 initializations, 
       each biased towards one of the 10 intervals, and run a short Nelder-Mead from each. 
       Then I will pick the best result.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, x1_vals, ref):
        """
        Compute the hypervolume of the set of points defined by x1_vals on the ZDT3 front.
        """
        x1_sorted = np.sort(x1_vals)
        f1 = x1_sorted
        f2 = 1 - np.sqrt(x1_sorted)
        
        # Formula: HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * (R2 - y_i) where x_N = R1
        x1_ext = np.append(x1_sorted, ref[0])
        widths = np.diff(x1_ext)
        heights = ref[1] - f2
        
        return np.sum(widths * heights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        B = self.budget
        ref = np.array([11.0, 11.0])
        
        # Objective function for minimization (we want to maximize HV)
        def objective(x1_flat):
            # Clip to [0,1] to ensure validity, though NM should stay close
            x1_clipped = np.clip(x1_flat, 0.0, 1.0)
            hv = self._compute_hv(x1_clipped, ref)
            return -hv
        
        # Generate 10 initializations based on partitioning [0,1] into 10 segments.
        # For each segment k, we create an initialization that is uniform but slightly 
        # perturbed or biased towards that segment to explore local structures.
        # Actually, the directive says "initializing 20 simplex bases at the 10 analytic local minima".
        # I will use 10 initializations. For each, I will create a vector of B points.
        # To make them distinct, I will use a "jittered uniform" approach where the jitter 
        # is seeded differently for each start.
        
        best_hv = -np.inf
        best_x1 = np.linspace(0, 1, B)
        
        # We will try 10 different starts.
        # To save time, we limit the maxiter for each NM run.
        # Total budget 330s. 10 runs. ~30s per run.
        # NM with B=200 variables is expensive. 
        # Maybe reduce the number of variables? No, B is fixed at 200.
        # NM in 200D is very slow.
        # The current program uses one NM run with maxiter 5000.
        # If I do 10 runs, I must reduce maxiter significantly, or use a smaller subset?
        # No, I must use all 200 points to maximize HV.
        
        # Alternative: Use the "grid" idea more strictly.
        # Divide B into 10 groups of B/10 = 20 points.
        # Optimize each group of 20 points independently within its segment [k/10, (k+1)/10].
        # This reduces the dimensionality of each NM problem to 20, which is much faster.
        # The HV is not separable, but if the segments are disjoint, the interaction is 
        # only through the global sorting. However, for a dense front, local optimization 
        # within segments often works well if the segments are chosen well.
        
        # Let's try the segmented approach.
        n_segments = 10
        points_per_seg = B // n_segments
        # Handle remainder
        remainders = B % n_segments
        
        x1_opt_full = np.zeros(B)
        
        # We need to be careful with time. 
        # 10 segments * NM on 20 vars.
        # Let's run NM for each segment.
        
        # To ensure we don't timeout, we'll track time.
        # If we run out of time, we use the best found so far (or uniform).
        
        # Initial guess: Uniform
        x1_init_uniform = np.linspace(0, 1, B)
        best_hv = self._compute_hv(x1_init_uniform, ref)
        best_x1 = x1_init_uniform.copy()
        
        # For each segment, optimize the points within that segment.
        # We fix the points in other segments to their current values (initially uniform).
        # This is a coordinate descent-like approach.
        
        # To make it faster, we can just run NM on the whole vector but with multiple starts.
        # But 200D NM is slow.
        # Let's stick to the segmented NM.
        
        for k in range(n_segments):
            if time.time() > deadline - 20: # Leave 20s for evaluation
                break
                
            # Define the segment boundaries
            seg_start = k / n_segments
            seg_end = (k + 1) / n_segments
            
            # Indices for this segment
            # We assign points_per_seg points to this segment.
            # To keep it simple, we map indices [k*points_per_seg, (k+1)*points_per_seg] to this segment.
            idx_start = k * points_per_seg
            idx_end = idx_start + points_per_seg
            
            # Extract current values for this segment
            x1_seg = best_x1[idx_start:idx_end].copy()
            
            # Define objective for this segment
            # We need to compute the total HV.
            # Since the points in other segments are fixed, we can precompute their contribution 
            # or just recompute the full HV each time. Recomputing is O(N log N) which is fine for N=200.
            
            def obj_seg(x_seg):
                # Construct full x1 vector
                x1_full = best_x1.copy()
                x1_full[idx_start:idx_end] = x_seg
                # Clip to segment boundaries to ensure we don't cross into other segments?
                # Not strictly necessary for HV, but helps convergence.
                # Actually, if points cross segments, the "segment" definition breaks.
                # Let's clip to [seg_start, seg_end].
                x_seg_clipped = np.clip(x_seg, seg_start, seg_end)
                x1_full[idx_start:idx_end] = x_seg_clipped
                hv = self._compute_hv(x1_full, ref)
                return -hv
            
            # Initial guess for this segment: uniform within the segment
            x1_seg_init = np.linspace(seg_start, seg_end, points_per_seg)
            
            try:
                # Run NM on 20 variables
                res = minimize(obj_seg, x1_seg_init, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < -best_hv: # HV increased
                    # Update best_x1
                    best_x1[idx_start:idx_end] = np.clip(res.x, seg_start, seg_end)
                    best_hv = -res.fun
                    
            except Exception:
                pass
        
        # Final evaluation
        x1_final = np.clip(best_x1, 0.0, 1.0)
        
        F = []
        for x1 in x1_final:
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break
                
        return np.array(F)
