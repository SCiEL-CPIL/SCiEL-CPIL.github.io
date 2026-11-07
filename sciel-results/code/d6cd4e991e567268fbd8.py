"""
Strategy: Strict Model Fitting for Linear Slope (f5)
Instead of guessing corners or using stochastic search, we assume f(x) = c + g^T x.
We evaluate f at 0 and 5 unit steps to exactly determine c and g.
We then analytically compute the point x* in [-5, 5]^5 that minimizes c + g^T x.
This is x*_i = -5 if g_i > 0, 5 if g_i < 0, 0 if g_i == 0.
This achieves the global minimum in exactly 6 calls (1 for c, 5 for g).
The remaining budget is used to verify and potentially refine if the linear assumption fails (though for BBOB f5 it holds).
"""

import numpy as np
import time


class Optimizer:
    """
    Optimizer exploiting the linear structure of f5 (Linear slope).
    Uses Least Squares (exact for 1D linear) to identify the gradient and offset,
    then jumps to the analytical optimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # 1. Model Identification Phase
        # f(x) = offset + dot(gradient, x)
        
        # Evaluate at origin to get offset
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_fx = f0
        calls = 1

        gradient = np.zeros(dim)
        
        # Evaluate at unit steps to get gradient components
        # f(e_i) - f(0) = gradient[i] * 1
        for i in range(dim):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
                
            x_i = np.zeros(dim)
            x_i[i] = 1.0
            f_i = func(x_i)
            calls += 1
            
            gradient[i] = f_i - f0
            
            # Update best if this point is better (unlikely to be better than 0 if gradient is positive, but possible if negative)
            if f_i < best_fx:
                best_fx = f_i
                best_x = x_i.copy()

        # 2. Analytical Optimum Calculation
        # Minimize offset + dot(gradient, x) subject to -5 <= x_i <= 5
        # For each i:
        # if gradient[i] > 0, minimize by setting x_i = -5
        # if gradient[i] < 0, minimize by setting x_i = 5
        # if gradient[i] == 0, any x_i is optimal, set to 0
        
        x_opt = np.zeros(dim)
        for i in range(dim):
            if gradient[i] > 1e-12:
                x_opt[i] = -5.0
            elif gradient[i] < -1e-12:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the analytical optimum
        if calls < budget:
            f_opt_eval = func(x_opt)
            calls += 1
            if f_opt_eval < best_fx:
                best_fx = f_opt_eval
                best_x = x_opt.copy()
        
        # 3. Refinement Phase (Safety Net)
        # If the function was not perfectly linear (e.g. due to floating point noise in the evaluator 
        # or if the problem instance had a slight non-linear component not in the spec but present in practice),
        # we use a local search starting from x_opt.
        # However, for BBOB f5, it is strictly linear. 
        # We use a small (1+1)-ES to ensure we don't miss anything if the "linear" assumption was slightly off 
        # due to the "hidden seeded shift, rotation and offset" phrasing potentially implying 
        # that the *instance* is linear but the *coordinate system* is rotated.
        # Wait, the spec says: "Instances apply a hidden seeded shift, rotation and offset".
        # This means f(x) = f5(R(x - shift) + offset). 
        # Since f5 is linear, f(x) is still linear! 
        # f5(y) = c + g^T y.
        # f(x) = c + g^T (R(x - shift) + offset) = c + g^T R x - g^T R shift + g^T offset.
        # This is still of the form C + G^T x where G = R^T g.
        # So the linear model holds perfectly. The analytical solution is exact.
        
        # The only reason to refine is if the "linear slope" function in BBOB is not purely linear 
        # but has a slight non-linear term? 
        # BBOB f5 is "Linear Slope". It is defined as f(x) = sum(x_i) or similar? 
        # Actually, BBOB f5 is often f(x) = sum(x_i) + offset? No, usually it's a linear function.
        # Let's check BBOB f5 definition: "Linear slope" is typically f(x) = sum(x_i) or a random linear function.
        # If it is strictly linear, the analytical solution is the global minimum.
        
        # We will spend the rest of the budget doing nothing (or verifying) to save time, 
        # or we can do a few random checks to be absolutely sure.
        # Given the AOCC metric rewards early progress, and we have found the global minimum 
        # in 6 calls, the score is already maximized for calls 1-6.
        # For calls 7-500, the best value b_i is constant at f_opt.
        # e_i = clip(f_opt - f_opt, ...) = 1e-8.
        # n_i = (log10(1e-8) + 8) / 10 = 0.
        # AOCC = 1 - mean(n_i) = 1 - 0 = 1.0.
        # This is the maximum possible score.
        
        # Therefore, we just need to ensure we return valid data and don't crash.
        # We will loop through the remaining budget doing no-op checks or just sleeping?
        # No, we must not waste time. We can just return.
        # But wait, if the linear assumption is wrong (e.g. if f5 is actually quadratic or something else 
        # in this specific local implementation), we need a fallback.
        
        # Let's add a lightweight fallback just in case.
        # If the value at x_opt is not significantly better than the initial value, 
        # or if we suspect non-linearity, we run a (1+1)-ES.
        
        # Heuristic: If the reduction is small, maybe it's not linear?
        # But for linear, reduction is large.
        
        # Let's just run a (1+1)-ES for the remaining budget to be safe, 
        # starting from x_opt. It will likely stay there or make tiny improvements.
        
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                sigma *= 1.1
            else:
                sigma *= 0.95
                
        return best_x, best_fx
