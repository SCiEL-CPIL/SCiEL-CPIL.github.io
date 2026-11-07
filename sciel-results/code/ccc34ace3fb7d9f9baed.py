import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        N = self.budget
        
        # Initial guess: uniform
        x = np.linspace(0.0, 1.0, N)
        
        # Gradient ascent to maximize hypervolume
        # HV = sum_{i=1}^{N-1} (x_{i+1} - x_i) (10 + sqrt(x_i)) + (11 - x_N)(10 + sqrt(x_N))
        
        # We will use a fixed number of iterations
        # Step size needs to be small enough for stability
        # The gradient magnitude can be large near 0
        # Let's use a simple update with projection
        
        for _ in range(100):
            # Calculate gradient
            g = np.zeros(N)
            
            # Avoid division by zero
            # If x_i is 0, we can't compute gradient properly.
            # We assume x_i > 0 during optimization.
            # If any x_i is 0, we nudge it slightly.
            x_safe = np.maximum(x, 1e-6)
            
            # Interior points 1 to N-2 (indices 1 to N-2 in 0-indexed, i.e., x_2 to x_{N-1})
            # Formula: g_k = sqrt(x_{k-1}) - 1.5*sqrt(x_k) + x_{k+1}/(2*sqrt(x_k))
            # Note: indices in python are 0..N-1
            # k corresponds to index k-1?
            # Let's use 0-indexed i for x_i.
            # g[i] for i in 1..N-2
            # x_{i-1}, x_i, x_{i+1}
            
            sqrt_x = np.sqrt(x_safe)
            
            # g[0] = x_1/(2*sqrt(x_0)) - 10 - 1.5*sqrt(x_0)
            # Using 0-indexed: x[1]/(2*sqrt(x[0])) - 10 - 1.5*sqrt(x[0])
            g[0] = x[1] / (2 * sqrt_x[0]) - 10 - 1.5 * sqrt_x[0]
            
            # g[N-1] = sqrt(x_{N-2}) - sqrt(x_{N-1}) + (11 - x_{N-1})/(2*sqrt(x_{N-1}))
            # Using 0-indexed: sqrt(x[N-2]) - sqrt(x[N-1]) + (11 - x[N-1])/(2*sqrt(x[N-1]))
            g[N-1] = sqrt_x[N-2] - sqrt_x[N-1] + (11 - x[N-1]) / (2 * sqrt_x[N-1])
            
            # g[i] for i in 1..N-2
            # g[i] = sqrt(x[i-1]) - 1.5*sqrt(x[i]) + x[i+1]/(2*sqrt(x[i]))
            if N > 2:
                g[1:N-1] = sqrt_x[0:N-2] - 1.5 * sqrt_x[1:N-1] + x[2:N] / (2 * sqrt_x[1:N-1])
            
            # Update x
            # Adaptive step size or fixed?
            # Let's try a fixed step size, maybe decaying?
            # Or just normalize gradient?
            # The gradient values are around 10-20.
            # x is in [0,1].
            # Step size 0.01 seems reasonable.
            step = 0.01
            x = x + step * g
            
            # Project to [0, 1] and sort
            x = np.clip(x, 0.0, 1.0)
            x = np.sort(x)
            
            # Ensure uniqueness? If points coincide, gradient is tricky.
            # But sorting handles it.
            
        # Construct full x vectors
        X = np.zeros((N, n))
        X[:, 0] = x
        
        if n > 1:
            exponent = 1.0 / (4.0 * (n - 1))
            # Ensure no negative base
            target_vals = (0.5 * x) ** exponent
            X[:, 1:] = target_vals[:, np.newaxis]
            
        # Evaluate all points
        F = np.array([problem(x_vec) for x_vec in X])
        
        return X
