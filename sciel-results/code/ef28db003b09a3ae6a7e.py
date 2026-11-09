import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        
        # Generate all 2^dim vertices of [-5, 5]^dim
        vertices = np.array([[v for v in [-5.0, 5.0]] for _ in range(dim)]).T if dim > 0 else np.array([[]])
        # Actually, let me generate properly
        import itertools
        verts = np.array(list(itertools.product([-5.0, 5.0], repeat=dim)))
        
        # Phase 1: Estimate linear model with a few evaluations
        # Start at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        best_x = x0.copy()
        best_f = f0
        
        # Estimate gradient using central differences with small step
        delta = 0.5
        grad = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += delta
            xm[i] -= delta
            xp = np.clip(xp, -5, 5)
            xm = np.clip(xm, -5, 5)
            fp = func(xp)
            fm = func(xm)
            calls += 2
            grad[i] = (fp - fm) / (2 * delta)
        
        # Phase 2: Rank vertices by predicted value and evaluate in order
        # Predicted value at vertex v: f0 + grad^T * v
        pred_values = f0 + verts @ grad
        
        # Sort vertices by predicted value (ascending = best first)
        order = np.argsort(pred_values)
        
        # Evaluate in order
        for idx in order:
            if calls >= budget or time.time() > deadline:
                break
            v = verts[idx]
            fv = func(v)
            calls += 1
            if fv < best_f:
                best_f = fv
                best_x = v.copy()
        
        # Phase 3: If we have budget left, do local refinement around best vertex
        # Actually, for a piecewise linear function, the minimum IS at a vertex,
        # so local refinement won't help. But let's use remaining budget wisely.
        
        # If we haven't evaluated all vertices, continue with remaining
        # Actually, we already iterate through all vertices in order, so this is covered.
        
        return best_x, best_f
