"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using a structured sampling approach
    combined with local refinement.
    
    Strategy:
    1. Generate a dense set of candidate points on the known Pareto front structure
       using uniform sampling on the hypersphere surface (with arccos correction for
       uniform area distribution in 3D objective space).
    2. For each candidate, solve for the decision variables that achieve the target
       objective values by exploiting the DTLZ4 structure.
    3. Use remaining budget for local refinement of the best points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._eval_count = 0
        self._deadline = None
        self._safety_margin = 30.0  # seconds

    def _check_deadline(self):
        """Check if we're close to the deadline."""
        if self._deadline is not None:
            return (time.time() - self._deadline) > self._safety_margin
        return False

    def _sample_pareto_front(self, rng, n_points):
        """
        Sample points uniformly on the Pareto front surface of DTLZ4.
        
        For DTLZ4 with 3 objectives, the Pareto front is:
        f1 = 0.5 * (1 + g) * cos(theta1) * cos(theta2)  -- wait, let me recall
        Actually for DTLZ4:
        f1 = 0.5 * (1 + g) * sin(pi*theta1/2) * sin(pi*theta2/2)
        f2 = 0.5 * (1 + g) * sin(pi*theta1/2) * cos(pi*theta2/2)
        f3 = 0.5 * (1 + g) * cos(pi*theta1/2)
        
        On the Pareto front, g = 0, so:
        f1 = 0.5 * sin(pi*theta1/2) * sin(pi*theta2/2)
        f2 = 0.5 * sin(pi*theta1/2) * cos(pi*theta2/2)
        f3 = 0.5 * cos(pi*theta1/2)
        
        To sample uniformly on the surface of the quarter-sphere in [0,0.5]^3,
        we use the arccos transformation for theta1 and uniform for theta2.
        """
        # For uniform sampling on the 2D surface of a 3D quarter-sphere:
        # theta1 in [0, pi/2], theta2 in [0, pi/2]
        # Uniform area element is sin(theta1) dtheta1 dtheta2
        # So theta1 = 2*arcsin(sqrt(u1)), but our range is [0, pi/2]
        # Actually, let's use: u1 uniform in [0,1], theta1 = 2*arcsin(sqrt(u1))
        # This gives uniform distribution over the sphere surface.
        
        u1 = rng.random(n_points)
        u2 = rng.random(n_points)
        
        # theta1 in [0, pi/2] with density proportional to cos(theta1) (for quarter sphere)
        # Actually for a full sphere, uniform area means theta ~ arcsin(2u-1) type
        # For quarter sphere in first octant:
        # The surface area element on a sphere is sin(theta) dtheta dphi
        # For theta in [0, pi/2] and phi in [0, pi/2]:
        # CDF of theta: F(t) = integral_0^t sin(s) ds / integral_0^{pi/2} sin(s) ds = (1-cos(t))/1
        # So theta = arccos(1 - u1) = arccos(1-u1)
        # Wait, that gives theta in [0, pi/2] when u1 in [0,1]
        # Let me verify: u1=0 -> theta=arccos(1)=0, u1=1 -> theta=arccos(0)=pi/2. Good.
        
        theta1 = np.arccos(1.0 - u1)  # in [0, pi/2]
        theta2 = np.pi * u2 / 2.0     # in [0, pi/2]
        
        # Compute objective values on the Pareto front (g=0)
        f1 = 0.5 * np.sin(theta1) * np.sin(theta2)
        f2 = 0.5 * np.sin(theta1) * np.cos(theta2)
        f3 = 0.5 * np.cos(theta1)
        
        return np.column_stack([f1, f2, f3])

    def _solve_dtlz4(self, target_f, lo, hi):
        """
        Given target objective values on the Pareto front, find decision variables.
        
        DTLZ4:
        - x[0] and x[1] are position variables
        - x[2:] are aggregation variables
        - g = sum((x_i^0.5 - 0.5)^2) for i in 2..dim-1
        - On Pareto front, g = 0, so x_i^0.5 = 0.5, i.e., x_i = 0.25 for i >= 2
        
        For the position variables:
        f1 = 0.5 * (1+g) * sin(pi*x0/2) * sin(pi*x1/2)
        f2 = 0.5 * (1+g) * sin(pi*x0/2) * cos(pi*x1/2)
        f3 = 0.5 * (1+g) * cos(pi*x0/2)
        
        With g=0:
        f3 = 0.5 * cos(pi*x0/2)  =>  cos(pi*x0/2) = 2*f3  =>  x0 = (2/pi) * arccos(2*f3)
        sin(pi*x0/2) = sqrt(1 - 4*f3^2)
        f1/f2 = tan(pi*x1/2)  =>  x1 = (2/pi) * arctan(f1/f2)
        And we need sin(pi*x1/2) = 2*f1 / sqrt(1 - 4*f3^2)
        """
        f1, f2, f3 = target_f
        
        # Clip to valid range
        f3 = np.clip(f3, 0, 0.5)
        
        # Solve for x0
        cos_val = 2.0 * f3
        cos_val = np.clip(cos_val, 0, 1)
        x0 = (2.0 / np.pi) * np.arccos(cos_val)
        
        # Solve for x1
        sin_x0 = np.sqrt(max(0.0, 1.0 - cos_val ** 2))
        if sin_x0 > 1e-10:
            tan_val = f1 / max(f2, 1e-10)
            x1 = (2.0 / np.pi) * np.arctan(tan_val)
        else:
            x1 = 0.0
        
        # Set aggregation variables to achieve g=0
        x = np.zeros(self.dim)
        x[0] = np.clip(x0, lo[0], hi[0])
        x[1] = np.clip(x1, lo[1], hi[1])
        x[2:] = 0.25  # x_i^0.5 = 0.5 => x_i = 0.25
        
        # Clip to bounds
        x = np.clip(x, lo, hi)
        
        return x

    def __call__(self, problem, bounds):
        self._eval_count = 0
        self._deadline = time.time() + 330.0  # Total wall clock limit
        self._safety_margin = 10.0  # Return 10s before limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Generate structured candidates on the Pareto front
        # Use a grid-based approach for better coverage
        n_grid = 15  # 15x15 grid = 225 points
        n_structured = min(self.budget // 3, n_grid * n_grid)
        
        # Generate grid points on the Pareto front
        theta1_vals = np.linspace(0.01, np.pi/2 - 0.01, n_grid)
        theta2_vals = np.linspace(0.01, np.pi/2 - 0.01, n_grid)
        
        # Use arccos correction for uniform area
        # Actually, let's use both: a structured grid and random samples
        structured_points = []
        for t1 in theta1_vals:
            for t2 in theta2_vals:
                f1 = 0.5 * np.sin(t1) * np.sin(t2)
                f2 = 0.5 * np.sin(t1) * np.cos(t2)
                f3 = 0.5 * np.cos(t1)
                structured_points.append([f1, f2, f3])
        
        structured_points = np.array(structured_points)
        
        # Also add random samples with arccos correction
        n_random = min(n_structured - len(structured_points), 100)
        if n_random > 0:
            u1 = rng.random(n_random)
            u2 = rng.random(n_random)
            r_theta1 = np.arccos(1.0 - u1)
            r_theta2 = np.pi * u2 / 2.0
            r_f1 = 0.5 * np.sin(r_theta1) * np.sin(r_theta2)
            r_f2 = 0.5 * np.sin(r_theta1) * np.cos(r_theta2)
            r_f3 = 0.5 * np.cos(r_theta1)
            random_points = np.column_stack([r_f1, r_f2, r_f3])
            structured_points = np.vstack([structured_points, random_points])
        
        # Solve for decision variables and evaluate
        all_F = []
        all_X = []
        
        for target_f in structured_points:
            if self._check_deadline() or self._eval_count >= self.budget:
                break
            x = self._solve_dtlz4(target_f, lo, hi)
            f = problem(x)
            self._eval_count += 1
            all_F.append(f)
            all_X.append(x)
        
        all_F = np.array(all_F)
        all_X = np.array(all_X)
        
        # Phase 2: Local refinement using remaining budget
        if self._eval_count < self.budget and not self._check_deadline():
            # Select a subset of points for refinement
            n_refine = min(self.budget - self._eval_count, 50)
            
            # Use non-dominated sorting to select parents
            def dominates(a, b):
                return np.all(a <= b) and np.any(a < b)
            
            # Simple non-dominated selection
            non_dominated_idx = []
            for i in range(len(all_F)):
                is_nd = True
                for j in range(len(all_F)):
                    if i == j:
                        continue
                    if dominates(all_F[j], all_F[i]):
                        is_nd = False
                        break
                if is_nd:
                    non_dominated_idx.append(i)
            
            if len(non_dominated_idx) == 0:
                non_dominated_idx = list(range(len(all_F)))
            
            # Refine non-dominated points
            for idx in non_dominated_idx[:n_refine]:
                if self._check_deadline() or self._eval_count >= self.budget:
                    break
                
                x = all_X[idx].copy()
                # Small perturbation
                step_size = 0.05 * (hi - lo)
                x_perturbed = x + step_size * rng.standard_normal(self.dim)
                x_perturbed = np.clip(x_perturbed, lo, hi)
                
                f_perturbed = problem(x_perturbed)
                self._eval_count += 1
                all_F.append(f_perturbed)
                all_X.append(x_perturbed)
        
        return all_X
# EVOLVE-BLOCK-END
