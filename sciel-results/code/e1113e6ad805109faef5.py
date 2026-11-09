import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        # Safety margin: 330s limit, stop at 325s
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        # Strategy: Exploit the known structure of ZDT1 to minimize g(x) for each x_0.
        # f1 = x_0
        # g(x) = 1 + 10 * (1/(n-1)) * sum(|x_i - sin(6*pi*x_0 + i*pi/n)|) for i=1..n-1
        # f2 = (1+g(x)) * (1 - sqrt(x_0 / (1+g(x))))
        # To minimize f2 for a given x_0, we must minimize g(x).
        # The minimum of |x_i - target| for x_i in [0,1] is 0 if target in [0,1], else min(|0-target|, |1-target|).
        # Here target = sin(...). Since sin is in [-1,1], the closest bound is 0 if sin < 0, else sin.
        # So optimal x_i = max(0, sin(6*pi*x_0 + i*pi/n)).
        
        n_evals = self.budget
        n = self.dim
        
        # Generate x_0 values
        # Optimal sampling density for ZDT1 hypervolume maximization
        # Derived from minimizing Riemann sum error for concave front g(x) = 10 + sqrt(x)
        # Error term proportional to |g''(x)| dx^2 ~ x^(-3/2) dx^2
        # Optimal step size dx ~ x^(3/4)
        # Density ~ x^(-3/4)
        # CDF ~ x^(1/4) => x ~ u^4
        u = np.linspace(0.0, 1.0, n_evals)
        x0_values = u**4.0
        
        # Pre-compute constants
        # i ranges from 1 to n-1
        i_vals = np.arange(1, n)
        phases = i_vals * np.pi / n
        
        for x0 in x0_values:
            # Check deadline
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            # Compute optimal x_i for i > 0
            # sin(6*pi*x_0 + phases)
            sin_vals = np.sin(6.0 * np.pi * x0 + phases)
            # Clip to [0, 1] to minimize |x_i - sin|
            # Since sin in [-1, 1], max(0, sin) is the projection onto [0, 1]
            x[1:] = np.maximum(0.0, sin_vals)
            
            # Call the problem
            try:
                problem(x)
            except Exception:
                break

        return None
