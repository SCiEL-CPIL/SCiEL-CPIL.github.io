import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0

def _autoconv_linear(x):
    n = len(x)
    size = 2 * n - 1
    X = np.fft.fft(x, size=size)
    ac = np.fft.ifft(X * np.conj(X)).real
    return ac

def _score(x):
    ac = _autoconv_linear(x)
    max_ac = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Continuous optimization to find good direction, then snap to signs
    x = np.ones(N).copy()
    
    # L-BFGS on continuous relaxation
    try:
        def obj(x):
            return _score(x)
        
        def obj_grad(x):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(N)
            for i in range(N):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
            return grad
        
        res = minimize(obj, x, method='L-BFGS-B', jac=obj_grad, 
                       options={'maxiter': 50, 'maxfun': 200})
        if time.time() - start_time < DEADLINE:
            x_cont = res.x
            # Snap to signs
            x_sign = np.sign(x_cont)
            x_sign[x_sign == 0] = 1
            s = _score(x_sign)
            if s < best_score:
                best_score = s
                best_x = x_sign.copy()
    except Exception:
        pass
    
    # Local search on ±1
    if time.time() - start_time < DEADLINE - 10:
        x = best_x.copy()
        for _ in range(10000):
            if time.time() - start_time > DEADLINE - 5:
                break
            i = np.random.randint(N)
            x[i] = -x[i]
            s = _score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
            else:
                x[i] = -x[i]
    
    return best_x
