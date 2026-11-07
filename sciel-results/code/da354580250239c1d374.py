import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Define the objective function to minimize (negative of the value)
    # We fix sum(h) = 1 to exploit scale invariance.
    # Then norm_1 = sum(g) = (sum(h))^2 = 1.
    # norm_inf = max(g).
    # value = l2 / (1 * max(g)) = l2 / max(g).
    
    def objective(h):
        h = np.asarray(h)
        h = np.maximum(h, 0)
        
        g = np.convolve(h, h)
        m = len(g)
        
        # Construct y
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        # Calculate l2
        # Sum_{i=0}^m (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
        # = 2 * sum(y^2) + sum(y[:-1] * y[1:])
        s1 = 2.0 * np.sum(y**2)
        s2 = np.sum(y[:-1] * y[1:])
        sum_val = s1 + s2
        
        w = 1.0 / (m + 1) # 0.01
        l2 = (w / 3.0) * sum_val
        
        # norm_1 is 1.0 because sum(h)=1 is enforced by constraint
        # norm_inf = max(g)
        norm_inf = np.max(g)
        
        if norm_inf < 1e-10:
            return 0.0
            
        val = l2 / norm_inf
        return -val

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    bounds = [(0.0, None) for _ in range(N)]
    
    # Initial guesses
    guesses = []
    x = np.linspace(-1, 1, N)
    
    # 1. Uniform
    guesses.append(np.ones(N) / N)
    
    # 2. Gaussian (narrow)
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0]:
        g1 = np.exp(-x**2 / (2 * sigma**2))
        guesses.append(g1 / np.sum(g1))
        
    # 3. Rectangular windows of different widths
    for width in [10, 20, 30, 40]:
        r = np.zeros(N)
        start = (N - width) // 2
        r[start:start+width] = 1.0
        guesses.append(r / np.sum(r))
        
    # 4. Triangular
    t = np.tri(N)
    guesses.append(t / np.sum(t))
    
    # 5. Random
    np.random.seed(42)
    r = np.random.rand(N)
    guesses.append(r / np.sum(r))
    
    best_val = -np.inf
    best_h = guesses[0]
    
    for h0 in guesses:
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            continue
            
    # Final cleanup
    best_h = np.maximum(best_h, 0)
    best_h /= np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
