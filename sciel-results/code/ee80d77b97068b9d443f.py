import time
import numpy as np
from scipy.optimize import brentq, newton

def find_root(f, x0):
    class CallCounter:
        def __init__(self, f):
            self.f = f
            self.calls = 0
            self.start_time = time.time()
            self.best_x = float(x0)
            self.best_fx = None
            try:
                self.best_fx = f(x0)
                self.calls += 1
            except:
                self.best_fx = float('inf')

        def __call__(self, x):
            self.calls += 1
            if time.time() - self.start_time > 110:
                raise TimeoutError()
            fx = self.f(x)
            if self.best_fx is None or abs(fx) < abs(self.best_fx):
                self.best_x = x
                self.best_fx = fx
            return fx

    counter = CallCounter(f)
    
    # 1. Try to find a bracket
    a, b = _find_bracket(counter, x0)
    
    if a is not None:
        try:
            root, res = brentq(counter, a, b, xtol=1e-12, rtol=1e-12, maxiter=100, full_output=True)
            if res.converged:
                return root
            else:
                return counter.best_x
        except TimeoutError:
            return counter.best_x
        except Exception:
            return counter.best_x
    else:
        # No bracket found, fall back to Newton
        try:
            root = newton(counter, x0, fprime=None, maxiter=100, tol=1e-12)
            return root
        except TimeoutError:
            return counter.best_x
        except Exception:
            return counter.best_x

def _find_bracket(counter, x0, max_expansions=50):
    # Initial check
    a = x0 - 1.0
    b = x0 + 1.0
    try:
        fa = counter(a)
        fb = counter(b)
    except TimeoutError:
        return None, None
        
    if fa * fb < 0:
        return a, b
        
    # Expand symmetrically
    for i in range(max_expansions):
        dist = 2.0 * (2.0 ** i)
        a = x0 - dist
        b = x0 + dist
        try:
            fa = counter(a)
            fb = counter(b)
        except TimeoutError:
            return None, None
            
        if fa * fb < 0:
            return a, b
            
    return None, None
