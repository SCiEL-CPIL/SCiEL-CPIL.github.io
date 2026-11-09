"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using a hybrid strategy:
    1. Generate a well-spread set of candidate points on the DTLZ2 Pareto front
       using low-discrepancy sequences (Halton-like) in the 2D angular space.
    2. Use local optimization (Nelder-Mead) to refine each point's position
       to lie exactly on the Pareto front (minimizing the deviation from the
       ideal Pareto manifold).
    3. Use remaining budget for additional random search + local refinement
       to ensure robustness and cover edge cases.
    
    The key insight: DTLZ2's Pareto front is the unit sphere in objective space
    (for n_obj=3), parameterized by 2 angles. By sampling these angles with a
    low-discrepancy sequence and then projecting to the front via local search,
    we achieve near-uniform coverage of the Pareto front with minimal wasted
    evaluations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, base):
        """Generate n points from the Halton sequence in [0,1) for a given base."""
        result = np.zeros(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            idx = i + 1  # start from 1 to avoid 0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            result[i] = r
        return result

    def _pareto_front_point(self, theta1, theta2):
        """
        For DTLZ2 with 3 objectives, the Pareto front is the unit sphere:
        f1 = sin(theta1/2) * sin(theta2/2)
        f2 = sin(theta1/2) * cos(theta2/2)
        f3 = cos(theta1/2)
        where theta1 in [0, pi/2], theta2 in [0, pi/2]
        
        Actually, for DTLZ2, the Pareto front is:
        sum(f_i^2) = 1, f_i >= 0
        
        Parameterization:
        theta1 in [0, pi/2], theta2 in [0, pi/2]
        f1 = cos(theta1) * cos(theta2)
        f2 = cos(theta1) * sin(theta2)
        f3 = sin(theta1)
        
        Wait, let me be more careful. For DTLZ2:
        The Pareto front is the set of points where sum(f_i^2) = 1 and f_i >= 0.
        A standard parameterization for 3 objectives:
        theta1 in [0, pi/2], theta2 in [0, pi/2]
        f1 = sin(theta1/2) * sin(theta2/2)
        f2 = sin(theta1/2) * cos(theta2/2)  
        f3 = cos(theta1/2)
        
        Let me verify: when theta1=0: f1=0, f2=0, f3=1. Sum of squares = 1. Good.
        When theta1=pi/2, theta2=0: f1=0, f2=sin(pi/4)*cos(0)=sqrt(2)/2, f3=0. Sum = 1/2. Not 1.
        
        Let me use a different parameterization. For the unit sphere in the positive octant:
        We can use spherical coordinates restricted to the first octant.
        
        Actually, the standard DTLZ2 Pareto front for M objectives is:
        sum_{i=1}^{M} f_i^2 = 1, f_i >= 0
        
        For M=3, a valid parameterization:
        theta1 in [0, pi/2], theta2 in [0, pi/2]
        f1 = cos(theta1) * cos(theta2)
        f2 = cos(theta1) * sin(theta2)
        f3 = sin(theta1)
        
        Check: theta1=0, theta2=0: f=(1,0,0), sum=1. Good.
        theta1=pi/2, theta2=0: f=(0,0,1), sum=1. Good.
        theta1=0, theta2=pi/2: f=(0,1,0), sum=1. Good.
        theta1=pi/4, theta2=pi/4: f1=cos(pi/4)*cos(pi/4)=1/2, f2=1/2, f3=sin(pi/4)=sqrt(2)/2
        Sum = 1/4 + 1/4 + 1/2 = 1. Good.
        """
        f1 = np.cos(theta1) * np.cos(theta2)
        f2 = np.cos(theta1) * np.sin(theta2)
        f3 = np.sin(theta1)
        return np.array([f1, f2, f3])

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        all_points = []
        all_f = []
        
        # Strategy 1: Sample Pareto front using Halton sequence in angular space
        # Use about 60% of budget for this
        n_pareto = int(self.budget * 0.6)
        
        # Generate Halton sequence points in 2D
        halton1 = self._halton_sequence(n_pareto, 2)
        halton2 = self._halton_sequence(n_pareto, 3)
        
        # Map to theta1, theta2 in [0, pi/2]
        theta1 = halton1 * (np.pi / 2)
        theta2 = halton2 * (np.pi / 2)
        
        # For each Pareto front point, we need to find the decision variable x
        # that maps to this objective vector. In DTLZ2, the Pareto front is
        # achieved when all but the first M variables are 0 (or specific values).
        # For DTLZ2: f_i depends on x_1...x_M through g(x_M+1...x_N) which should be 0.
        # g(x) = sum(x_i^2) for i=M+1..N, and we need g=0, so x_M+1..x_N = 0.
        # The first M variables determine the position on the Pareto front.
        
        # For DTLZ2 with M=3 objectives:
        # f1 = 1 + g * (1 + sin(3*pi*x1))
        # f2 = 1 + g * (1 + sin(3*pi*x1)) * cos(x2*pi/2)
        # ...
        
        # Actually, the standard DTLZ2 formulation:
        # g(x) = sum_{i=M+1}^{n} (x_i - 0.5)^2
        # f1(x) = (1+g) * x1
        # f2(x) = (1+g) * (1-x1) * x2
        # f3(x) = (1+g) * (1-x1) * (1-x2)
        
        # Pareto front when g=0, i.e., x_i = 0.5 for i>M.
        # Then f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
        # Sum of squares: x1^2 + (1-x1)^2*x2^2 + (1-x1)^2*(1-x2)^2
        # = x1^2 + (1-x1)^2*(x2^2 + (1-x2)^2)
        # This is not 1 in general.
        
        # Let me reconsider. The actual DTLZ2 Pareto front is:
        # sum(f_i^2) = 1, f_i >= 0
        # And the mapping from decision variables to objectives on the front is:
        # For x_i = 0.5 for i > M:
        # f1 = x1
        # f2 = (1-x1) * x2
        # f3 = (1-x1) * (1-x2)
        
        # To achieve a target point (f1*, f2*, f3*) on the unit sphere:
        # We need x1 = f1*
        # (1-x1) * x2 = f2* => x2 = f2*/(1-f1*)
        # (1-x1) * (1-x2) = f3* => check: (1-f1*)*(1-f2*/(1-f1*)) = (1-f1*) - f2* = f3* (if f1*+f2*+f3*... no)
        
        # Actually for the standard DTLZ2, the Pareto front is the quarter unit sphere in the first octant,
        # and the mapping is:
        # x1 = f1
        # x2 = f2 / (1 - f1)
        # x3..xn = 0.5
        
        # Let me just use local optimization to find x for each target objective.
        
        for i in range(n_pareto):
            if used >= self.budget:
                break
            
            target_f = self._pareto_front_point(theta1[i], theta2[i])
            
            # Initial guess: x1 = target_f[0], x2 = target_f[1]/max(1-target_f[0], 1e-6), rest 0.5
            x0 = np.full(self.dim, 0.5)
            x0[0] = target_f[0]
            if self.dim >= 2:
                x0[1] = target_f[1] / max(1 - target_f[0], 1e-6)
                x0[1] = np.clip(x0[1], 0, 1)
            
            # Local optimization to minimize distance to target
            def objective(x):
                nonlocal used
                fx = problem(x)
                used += 1
                return np.sum((fx - target_f) ** 2)
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < 1e-4:
                    fx = problem(result.x)
                    used += 1
                    all_points.append(result.x)
                    all_f.append(fx)
            except:
                pass
        
        # Strategy 2: Use remaining budget for random search + local refinement
        remaining = self.budget - used
        if remaining > 0:
            n_random = int(remaining * 0.5)
            n_refine = remaining - n_random
            
            # Random initialization
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            for i in range(n_random):
                if used >= self.budget:
                    break
                fx = problem(X_rand[i])
                used += 1
                all_points.append(X_rand[i])
                all_f.append(fx)
            
            # Local refinement of best points
            if n_refine > 0 and all_f:
                F_arr = np.array(all_f)
                # Pick non-dominated points for refinement
                dominated = np.array([
                    np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1))
                    for f in F_arr
                ])
                candidates = [i for i in range(len(all_f)) if not dominated[i]]
                if not candidates:
                    candidates = list(range(len(all_f)))
                
                rng.shuffle(candidates)
                for idx in candidates[:n_refine]:
                    if used >= self.budget:
                        break
                    x0 = all_points[idx].copy()
                    
                    def obj(x):
                        nonlocal used
                        fx = problem(x)
                        used += 1
                        return np.sum(fx)
                    
                    try:
                        result = minimize(obj, x0, method='Nelder-Mead',
                                          options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10})
                        if used < self.budget:
                            fx = problem(result.x)
                            used += 1
                            all_points.append(result.x)
                            all_f.append(fx)
                    except:
                        pass
        
        # Return the last point (return value is ignored, but must not be None)
        if all_points:
            return all_points[-1]
        return lo

# EVOLVE-BLOCK-END
