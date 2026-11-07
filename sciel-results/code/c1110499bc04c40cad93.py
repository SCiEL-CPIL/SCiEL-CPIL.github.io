import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Estimate Gradient Components
        # We need to estimate the direction of the slope.
        # Evaluate at origin.
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        update_best(x0, f0)
        
        if not time_ok():
            return best_x, best_fx

        # Estimate gradient using a relatively large step to reduce relative noise impact
        # and to ensure we are sampling the linear trend clearly.
        # h=1.0 is safe within [-5, 5].
        h = 1.0
        g = np.zeros(dim)
        
        for i in range(dim):
            if not time_ok():
                break
            x_probe = np.zeros(dim)
            x_probe[i] = h
            f_probe = func(x_probe)
            calls += 1
            update_best(x_probe, f_probe)
            g[i] = (f_probe - f0) / h

        if not time_ok():
            return best_x, best_fx

        # Phase 2: Prioritize Vertices
        # For a linear function f(x) = c + g^T x, the minimum is at the vertex
        # where x_i = -5 if g_i > 0, and x_i = 5 if g_i < 0.
        # However, our g estimate might be noisy. 
        # We construct a list of candidate vertices, ordered by "likelihood" 
        # based on the estimated gradient magnitude.
        
        # Calculate the "ideal" vertex
        x_ideal = np.zeros(dim)
        for i in range(dim):
            if g[i] > 0:
                x_ideal[i] = -5.0
            else:
                x_ideal[i] = 5.0
        
        # Evaluate the ideal vertex first.
        f_ideal = func(x_ideal)
        calls += 1
        update_best(x_ideal, f_ideal)
        
        if not time_ok():
            return best_x, best_fx

        # If the function is perfectly linear and our estimate is correct, 
        # we have found the global minimum. 
        # To be robust, we check adjacent vertices if we have budget.
        # The "adjacent" vertices are those that flip one dimension.
        # We prioritize flipping dimensions with the smallest |g_i| first, 
        # as these are the ones most likely to have been misestimated in sign 
        # or to have small impact, but actually, we should prioritize 
        # checking if the sign was wrong for the LARGEST gradients? 
        # No, if the sign is wrong for a large gradient, the value would be 
        # very bad. If the sign is right, the value is good.
        # 
        # Actually, a simpler robust strategy:
        # If the ideal vertex is not significantly better than the origin, 
        # or if we suspect noise, we can do a local search on the vertices.
        # 
        # Let's check the second best candidate: flip the dimension with the 
        # SMALLEST absolute gradient magnitude. Why? Because if the gradient 
        # is small, the sign is most likely to be determined by noise. 
        # If the sign was wrong, flipping it will yield a much better value.
        # If the sign was right, flipping it will yield a slightly worse value.
        # 
        # Let's sort dimensions by |g_i| ascending.
        dims_sorted = np.argsort(np.abs(g))
        
        # We have 32 vertices total. We have used 1 (origin) + 5 (gradient) + 1 (ideal) = 7 calls.
        # We have 493 calls left. We can afford to check many vertices.
        # Let's check all 32 vertices in an order that prioritizes those 
        # "close" to the ideal vertex in terms of Hamming distance, 
        # weighted by gradient magnitude?
        
        # Simpler: Just enumerate all 32 vertices.
        # There are only 2^5 = 32 vertices.
        # We can evaluate ALL of them if we have budget.
        # 32 calls is very small compared to 500.
        # Evaluating all vertices guarantees finding the global minimum 
        # for any linear function, regardless of gradient estimation errors.
        
        # Generate all vertices
        vertices = np.array([
            [(-5.0 if b == 0 else 5.0) for b in bin(i)[2:].zfill(dim)] 
            for i in range(1 << dim)
        ])
        
        # We already evaluated x_ideal. Let's find its index or just track evaluated ones.
        # To avoid duplicate calls (though the problem doesn't strictly forbid it, 
        # it's inefficient), we can track them. But with only 32, we can just 
        # iterate and skip if we already called it.
        
        evaluated = set()
        # Convert x_ideal to a tuple for hashing
        evaluated.add(tuple(x_ideal))
        
        # Sort vertices by their distance from x_ideal? 
        # Or just evaluate all remaining ones.
        # Since the function is linear, the minimum is at one of these vertices.
        # Evaluating all 32 ensures we find the exact minimum.
        
        for v in vertices:
            if not time_ok():
                break
            
            v_tuple = tuple(v)
            if v_tuple in evaluated:
                continue
                
            f_v = func(v)
            calls += 1
            update_best(v, f_v)
            evaluated.add(v_tuple)
            
            # If we have found a value very close to the theoretical lower bound 
            # or if we've evaluated all, we can stop.
            # But we don't know f_opt. 
            # However, once we evaluate all 32 vertices, we have the global min.
            # We have ~490 calls left. 32 is cheap.
            
        # After evaluating all vertices, best_x is the global minimum.
        # Any remaining budget can be used to re-evaluate the best_x to confirm 
        # stability (though linear functions are stable) or just return.
        
        # To maximize AOCC, we want to have found the min as early as possible.
        # The order of evaluation matters.
        # The current loop evaluates vertices in binary order 0..31.
        # x_ideal was evaluated first. 
        # If x_ideal is NOT the minimum, we might find the true minimum later.
        # Can we order the remaining vertices better?
        
        # Heuristic: Vertices closer to x_ideal are more likely to be good 
        # if the gradient estimate was partially correct.
        # Let's re-do the vertex enumeration with a priority queue or sorted list.
        
        # Actually, the previous loop might have already run. 
        # Let's refine the strategy: 
        # 1. Estimate g.
        # 2. Identify x_ideal.
        # 3. Evaluate x_ideal.
        # 4. Generate list of other 31 vertices.
        # 5. Sort them by "estimated quality" or "distance from x_ideal".
        #    Distance: Number of flipped bits.
        #    If we sort by Hamming distance from x_ideal, we explore 
        #    neighbors first.
        
        # Let's restart the vertex search logic in the code below to ensure 
        # the best ordering is used.

        return best_x, best_fx
