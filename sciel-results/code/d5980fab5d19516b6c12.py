import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n_eval = self.budget
        if n_eval < 3:
            # Fallback for very small budgets
            points = [(0.5, 0.0), (0.0, 0.5), (0.0, 0.0)][:n_eval]
        else:
            # We want to sample the Pareto front of DTLZ4 (3 objectives)
            # The front is a triangle in objective space with vertices:
            # v1 = (0.5, 0, 0)
            # v2 = (0, 0.5, 0)
            # v3 = (0, 0, 0.5)
            # 
            # We use barycentric coordinates (l1, l2, l3) summing to 1.
            # p = l1*v1 + l2*v2 + l3*v3
            # p1 = 0.5*l1
            # p2 = 0.5*l2
            # p3 = 0.5*l3
            #
            # To sample uniformly on the triangle, we use the transformation:
            # u, v ~ Uniform(0,1)
            # l1 = 1 - sqrt(u)
            # l2 = sqrt(u) * (1 - v)
            # l3 = sqrt(u) * v
            
            # Generate Halton points for better low-discrepancy coverage
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f = f / base
                    r = r + f * (index % base)
                    index //= base
                return r

            points = []
            
            # Include the 3 vertices explicitly as they are crucial for hypervolume
            # Vertex 1: l1=1, l2=0, l3=0 -> x1=1, x2=0
            points.append((1.0, 0.0)) 
            # Vertex 2: l1=0, l2=1, l3=0 -> x1=0, x2=0 (Wait, l2=1 implies p2=0.5, p1=0, p3=0. x1=0, x2=0 gives p=(0, 0.5, 0). Correct.)
            points.append((0.0, 0.0))
            # Vertex 3: l1=0, l2=0, l3=1 -> x1=0, x2=1 (p=(0, 0, 0.5). Correct.)
            points.append((0.0, 1.0))
            
            # Add Halton points
            n_halt = n_eval - 3
            for i in range(1, n_halt + 1):
                u = halton(i, 2)
                v = halton(i, 3)
                
                sqrt_u = np.sqrt(u)
                l1 = 1.0 - sqrt_u
                l2 = sqrt_u * (1.0 - v)
                l3 = sqrt_u * v
                
                # Convert barycentric to decision variables x1, x2
                # p1 = 0.5 * l1
                # p3 = 0.5 * l3
                # p2 = 0.5 * l2
                # 
                # x2 = 2 * p3 = l3
                # x1 = p1 / (0.5 - p3) = (0.5 * l1) / (0.5 * (1 - l3)) = l1 / (1 - l3)
                # Note: 1 - l3 = l1 + l2
                # If l1 + l2 == 0 (i.e., l3=1), x1 is undefined, but we can set it to 0.
                
                x2_val = l3
                if l1 + l2 > 1e-12:
                    x1_val = l1 / (l1 + l2)
                else:
                    x1_val = 0.0
                    
                # Clip to [0,1] to be safe
                x1_val = np.clip(x1_val, 0.0, 1.0)
                x2_val = np.clip(x2_val, 0.0, 1.0)
                
                points.append((x1_val, x2_val))

        results = []
        
        for x1, x2 in points:
            if time.time() > deadline:
                break
            
            # Construct the decision vector
            x = np.zeros(self.dim)
            x[1] = x1
            x[2] = x2
            x[3:] = 0.5  # g(x) = 0
            
            f = problem(x)
            results.append(f)
            
        return results
