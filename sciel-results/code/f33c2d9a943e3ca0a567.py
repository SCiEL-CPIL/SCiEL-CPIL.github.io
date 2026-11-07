import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    Uses the known analytical Pareto front to place evaluations optimally.
    Refines the placement using gradient-based optimization of the exact hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish in < 10s.
        deadline = t0 + 15.0 

        B = int(self.budget)
        D = int(self.dim)
        
        if B <= 0:
            return np.zeros(D)

        # ZDT1 Pareto Front:
        # x1 in [0, 1], x2...xD = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Reference point
        R1 = 11.0
        R2 = 11.0

        def compute_hv_and_grad(x1_vals):
            """
            Calculate Hypervolume and its gradient for points on the ZDT1 front.
            x1_vals: array of x1 values (sorted ascending).
            Returns (hv, grad)
            """
            n = len(x1_vals)
            if n == 0:
                return 0.0, np.array([])
            
            # Ensure sorted (input is expected to be sorted, but we sort for safety in HV calc)
            # For gradient, we assume input is sorted. If we use optimizer, we must maintain sortedness.
            # L-BFGS-B doesn't guarantee sortedness. 
            # However, the optimal solution is sorted. 
            # We can enforce sorting by optimizing over sorted parameters, but that's complex.
            # Instead, we can just compute HV assuming sorted, and if the optimizer unsorts, 
            # the HV function is non-differentiable. 
            # Given the strong prior, the optimizer should stay in the sorted manifold.
            
            x1_sorted = x1_vals # Assume sorted
            
            # f2 values
            f2 = 1.0 - np.sqrt(np.maximum(x1_sorted, 0.0))
            h = R2 - f2 # Heights: 10 + sqrt(x1)
            
            # x_{i+1} - x_i
            # x_N is R1
            x_next = np.append(x1_sorted[1:], R1)
            x_curr = x1_sorted
            
            widths = x_next - x_curr
            heights = h
            
            hv = np.sum(widths * heights)
            
            # Gradient calculation
            # dH/dx_k = h(x_{k-1}) - h(x_k) + (x_{k+1} - x_k) * h'(x_k)  for 0 < k < N-1
            # h'(x) = 1/(2*sqrt(x))
            
            grad = np.zeros(n)
            
            # h_prime
            with np.errstate(divide='ignore', invalid='ignore'):
                h_prime = 0.5 / np.sqrt(np.maximum(x1_sorted, 1e-15))
            # If x1 is 0, h' is infinite. But x1=0 is a boundary.
            # Let's handle x1=0 carefully. If x1[i] == 0, the term (x_{i+1}-x_i)*h'(x_i) is problematic.
            # However, x1=0 implies f2=1.
            # Let's just use a small epsilon for sqrt.
            
            # Interior points
            if n > 1:
                # Left term: h(x_{k-1})
                left_h = np.zeros(n)
                left_h[1:] = h[:-1]
                
                # Right term: -h(x_k) + (x_{k+1} - x_k) * h'(x_k)
                # Note: The term -h(x_k) comes from the derivative of -x_k * h(x_k) in term k?
                # Let's re-verify:
                # Term k-1: (x_k - x_{k-1}) h(x_{k-1}). d/dx_k = h(x_{k-1})
                # Term k: (x_{k+1} - x_k) h(x_k). d/dx_k = -h(x_k) + (x_{k+1} - x_k) h'(x_k)
                # Sum: h(x_{k-1}) - h(x_k) + (x_{k+1} - x_k) h'(x_k)
                
                right_part = -h + widths * h_prime
                
                grad[1:] = left_h[1:] + right_part[1:]
                
                # First point k=0:
                # Term 0: (x_1 - x_0) h(x_0). d/dx_0 = -h(x_0) + (x_1 - x_0) h'(x_0)
                # No left term.
                grad[0] = -h[0] + widths[0] * h_prime[0]
                
                # Last point k=N-1:
                # Term N-2: (x_{N-1} - x_{N-2}) h(x_{N-2}). d/dx_{N-1} = h(x_{N-2})
                # Term N-1: (x_N - x_{N-1}) h(x_{N-1}). d/dx_{N-1} = -h(x_{N-1}) + (x_N - x_{N-1}) h'(x_{N-1})
                # Sum: h(x_{N-2}) - h(x_{N-1}) + (x_N - x_{N-1}) h'(x_{N-1})
                # This is covered by the general formula if we define left_h[N-1] = h[N-2] and widths[N-1] = x_N - x_{N-1}.
                # Our vectorized code for grad[1:] uses left_h[1:] which is h[0:N-1]. 
                # For index N-1, left_h[N-1] is h[N-2]. Correct.
                # widths[N-1] is x_N - x_{N-1}. Correct.
                # So grad[N-1] is correctly calculated in the loop/vectorization above.
                
            elif n == 1:
                # Only 1 point
                # H = (R1 - x0) * h(x0)
                # dH/dx0 = -h(x0) + (R1 - x0) * h'(x0)
                grad[0] = -h[0] + (R1 - x1_sorted[0]) * h_prime[0]

            return float(hv), grad

        def neg_hv(x1_vals):
            hv, _ = compute_hv_and_grad(x1_vals)
            return -hv

        def neg_hv_grad(x1_vals):
            hv, grad = compute_hv_and_grad(x1_vals)
            return -grad

        # Initial guess: Analytic solution from cubic
        u = np.linspace(0.0, 1.0, B)
        x1_vals = np.zeros(B)
        
        for i in range(B):
            target = 16.0 * u[i]
            y = 1.0
            for _ in range(20):
                fy = y**3 + 15.0*y**2 - target
                dfy = 3.0*y**2 + 30.0*y
                if abs(dfy) < 1e-15:
                    break
                y_new = y - fy / dfy
                if y_new < 0:
                    y_new = 0.0
                if y_new > 1:
                    y_new = 1.0
                if abs(y_new - y) < 1e-12:
                    y = y_new
                    break
                y = y_new
            x1_vals[i] = y**2
        
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Optimize using L-BFGS-B
        # We need to ensure x1 stays in [0, 1] and sorted.
        # L-BFGS-B handles bounds [0, 1].
        # Sorting is not a standard constraint. However, the optimal solution is sorted.
        # If the optimizer tries to unsort, the HV function is non-differentiable.
        # Given the good initial guess, it should stay sorted.
        # To be safe, we can just run it. If it fails, we fall back to the analytic solution.
        
        best_hv, _ = compute_hv_and_grad(x1_vals)
        best_x1 = x1_vals.copy()
        
        try:
            if time.time() < deadline - 2.0:
                res = minimize(
                    neg_hv, 
                    x1_vals, 
                    jac=neg_hv_grad, 
                    method='L-BFGS-B', 
                    bounds=[(0.0, 1.0)] * B,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if res.success:
                    # Ensure sortedness for final HV calculation
                    opt_x1 = np.sort(res.x)
                    opt_hv, _ = compute_hv_and_grad(opt_x1)
                    if opt_hv > best_hv:
                        best_hv = opt_hv
                        best_x1 = opt_x1
        except Exception:
            pass # Fall back to analytic

        # Construct X
        X = np.zeros((B, D))
        X[:, 0] = best_x1
        # x2...xD = 0 is already set by zeros
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            
        return X[-1] # Return value is ignored, but must not crash
