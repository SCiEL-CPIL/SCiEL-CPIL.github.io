"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate Descent with Adaptive Step Sizes and Golden Section Search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        self.func = func
        self.cache = {}
        self.calls = 0
        
        # Use a fixed seed for reproducibility in case of ties or specific behaviors
        rng = np.random.default_rng(0)
        
        # Initialize x
        # Starting at 0 is often good for BBOB as shifts are not huge, 
        # but random is safer against local minima if any (though f12 is unimodal).
        # Let's start at 0 to be close to the likely center of the valley.
        x = np.zeros(self.dim)
        
        # Evaluate initial point
        fx = self._eval(x)
        
        # Initialize step sizes
        # Start with a step size that covers a significant portion of the box
        h = np.ones(self.dim) * 1.0
        
        # Golden section constant
        gr = (np.sqrt(5) - 1) / 2  # 0.618...
        
        # Main loop
        while self.calls < self.budget:
            improved = False
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                # Define 1D function for dimension i
                def f1d(t, i=i, x=x):
                    x_temp = x.copy()
                    x_temp[i] = t
                    return self._eval(x_temp)
                
                # Search interval
                a = x[i] - h[i]
                b = x[i] + h[i]
                
                # Clip to box bounds
                a = max(a, -5.0)
                b = min(b, 5.0)
                
                if b <= a:
                    # If interval is invalid, shrink h
                    h[i] = max(h[i] * 0.5, 1e-6)
                    continue
                
                # Perform Golden Section Search
                # We limit the number of steps to save budget
                # 3 steps per axis per cycle
                c = a + (1 - gr) * (b - a)
                d = a + gr * (b - a)
                
                fc = f1d(c)
                fd = f1d(d)
                
                # Iterative refinement
                # We do 2 more iterations to narrow down
                for _ in range(2):
                    if self.calls >= self.budget:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = a + (1 - gr) * (b - a)
                        fc = f1d(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = f1d(d)
                
                # Determine best point in the interval
                if fc < fd:
                    best_t = c
                    best_f = fc
                else:
                    best_t = d
                    best_f = fd
                
                # Update x and fx
                if best_f < fx:
                    x[i] = best_t
                    fx = best_f
                    improved = True
                    
                    # Adapt step size
                    # If the best point is near the edge, the minimum might be further
                    if best_t <= a + 1e-5 or best_t >= b - 1e-5:
                        h[i] = min(h[i] * 1.5, 10.0)
                    else:
                        # If inside, we found a local min, shrink to refine
                        h[i] = max(h[i] * 0.5, 1e-6)
                else:
                    # No improvement, shrink to refine
                    h[i] = max(h[i] * 0.5, 1e-6)
            
            # If no dimension improved, shrink all steps
            if not improved:
                h = np.maximum(h * 0.5, 1e-6)
                
        return x, fx

    def _eval(self, x):
        x = np.clip(x, -5, 5)
        # Use rounded tuple as key to handle float precision issues
        key = tuple(np.round(x, 10))
        if key in self.cache:
            return self.cache[key]
        
        val = self.func(x)
        self.cache[key] = val
        self.calls += 1
        return val
# EVOLVE-BLOCK-END
