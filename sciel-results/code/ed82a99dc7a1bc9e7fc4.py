"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution with Local Refinement for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        # Wrap the function to count calls, track best, and enforce budget
        def counting_func(x):
            self._call_count += 1
            if self._call_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            
            if f_val < self._best_f:
                self._best_f = f_val
                self._best_x = x.copy()
                
            return f_val

        bounds = [(-5.0, 5.0) for _ in range(dim)]

        # Phase 1: Global Search using Differential Evolution
        # We reserve some budget for local refinement.
        # DE population size is typically 15*dim. For dim=5, pop=75.
        # Each iteration evaluates pop size + 1 (or so) candidates.
        # With 500 budget, we can afford roughly 500/75 ~ 6-7 iterations if we use all budget.
        # But we want to save budget for local search.
        # Let's aim for ~300-400 evaluations in DE.
        
        # Set maxiter to a high number and let the budget exception stop it.
        # However, DE in scipy might do extra evaluations (polish, etc.).
        # We will use a callback or just rely on the exception.
        
        try:
            # tol and maxiter control convergence. 
            # We want to run until budget is nearly exhausted or it converges.
            # maxiter=100 is a safe upper bound for the number of iterations, 
            # but the exception will stop it earlier if budget hits.
            differential_evolution(
                counting_func,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=False,  # We will do our own polish to control budget strictly
                init='sobol'
            )
        except ValueError:
            # Budget exceeded, stop DE
            pass
        except Exception:
            # Other errors, stop DE
            pass

        # Phase 2: Local Refinement
        # Use the best point found by DE.
        # If DE didn't find anything (unlikely), use origin.
        if self._best_x is None:
            self._best_x = np.zeros(dim)
            try:
                self._best_f = counting_func(self._best_x)
            except ValueError:
                pass
            except Exception:
                pass

        # Use L-BFGS-B for local refinement. It is efficient and gradient-based (numerical).
        # It converges quickly to local minima.
        if self._call_count < budget:
            try:
                minimize(
                    counting_func,
                    self._best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'gtol': 1e-8}
                )
            except ValueError:
                pass
            except Exception:
                pass
        
        # Phase 3: Final Polish with Nelder-Mead if budget remains
        # Nelder-Mead is derivative-free and can sometimes find better local minima 
        # if L-BFGS-B got stuck in a flat region or numerical gradient issue.
        if self._call_count < budget - 10:
            try:
                minimize(
                    counting_func,
                    self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except ValueError:
                pass
            except Exception:
                pass

        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
