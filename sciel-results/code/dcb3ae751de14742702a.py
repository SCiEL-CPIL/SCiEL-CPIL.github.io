"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing


class Optimizer:
    """Dual Annealing based optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._call_count = 0

        # Wrap the function to count calls and clip inputs
        def wrapped(x):
            self._call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return func(x)

        # Define bounds
        bounds = [(-5.0, 5.0) for _ in range(dim)]

        # We need to limit the number of function evaluations.
        # dual_annealing does not have a direct maxfev parameter in all versions,
        # but we can use a custom termination or just let it run and hope it converges fast.
        # However, the contract says >500 calls is invalid.
        # We must strictly enforce the budget.
        # scipy.optimize.dual_annealing has a 'maxiter' and 'initial_temp', etc.
        # It doesn't have a hard 'maxfev' stop that raises an exception cleanly in all cases without wrapping.
        # Let's use a different approach: a custom multi-start local search or a specific scipy method.
        
        # Actually, `scipy.optimize.differential_evolution` has `maxiter` and `polish`.
        # But DE is stochastic.
        
        # Let's stick to a robust approach: 
        # 1. A few random starts.
        # 2. L-BFGS-B or Nelder-Mead from each start.
        # This is a classic "multi-start local search" which is different from (1+1)-ES.
        
        # The current program uses (1+1)-ES. 
        # Let's use Multi-Start Nelder-Mead or L-BFGS-B.
        # Nelder-Mead is derivative-free and robust.
        
        # Strategy:
        # - Use 50% of budget for multiple Nelder-Mead runs from different starts.
        # - Use the best result found so far.
        
        # To strictly control the budget, we will implement our own loop calling scipy.optimize.minimize
        # with a callback to check the budget, or simply estimate the number of iterations.
        
        # A safer bet: Use `scipy.optimize.minimize` with method='Nelder-Mead'.
        # We can pass a `callback` that raises an exception or returns True to stop if budget is exceeded.
        # However, exceptions might be caught by scipy.
        
        # Let's try a different strategy: 
        # Since f23 is multimodal, a global optimizer like `differential_evolution` might be better, 
        # but it's slow. 
        # `dual_annealing` is designed for global optimization.
        
        # Let's try to use `dual_annealing` but wrap the function to count calls and raise an exception 
        # if budget is exceeded, then catch it. But the contract says "an exception later keeps the calls made".
        # If we raise an exception, the remaining calls are lost, but the previous ones count.
        # The return value of __call__ is ignored.
        
        # However, if we raise an exception, does the evaluator mark it invalid? 
        # "More than 500 calls, or an exception before the first call, makes the whole result invalid; an exception later keeps the calls made."
        # So if I raise an exception after call 501, it's invalid. 
        # If I raise an exception after call 499, it keeps the 499 calls.
        # So I can use an exception to stop the optimizer when I hit the budget.
        
        # Let's implement a wrapper that raises a custom exception when budget is exceeded.
        
        class BudgetExceeded(Exception):
            pass

        def counting_func(x):
            self._call_count += 1
            if self._call_count > budget:
                raise BudgetExceeded()
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return func(x)

        best_x = None
        best_f = np.inf

        # Try dual_annealing
        try:
            res = dual_annealing(
                counting_func,
                bounds=bounds,
                initial_temp=5210.5,
                restart_temp_ratio=2e-5,
                maxiter=100,
                verbose=0
            )
            # dual_annealing might not stop exactly at budget, but the exception will catch it.
            # If it finishes naturally, we have a result.
            if res.success or hasattr(res, 'fun'):
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # If dual_annealing didn't use much budget, or failed, try Nelder-Mead from the best point or random starts
        # But we must be careful with the budget.
        
        # Let's add a fallback: Nelder-Mead from the best point found, or a random point if none.
        if best_x is None:
            # If no result, pick a random point
            best_x = np.random.uniform(-5, 5, dim)
            try:
                best_f = counting_func(best_x)
            except BudgetExceeded:
                pass
            except Exception:
                pass

        # Use remaining budget for local refinement with Nelder-Mead
        if self._call_count < budget:
            try:
                # Nelder-Mead can be sensitive to initial simplex size
                # We'll just let scipy handle it
                res2 = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res2.success or hasattr(res2, 'fun'):
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
