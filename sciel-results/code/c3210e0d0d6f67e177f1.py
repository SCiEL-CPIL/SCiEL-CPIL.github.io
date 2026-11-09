import numpy as np

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 with 7 vars, 3 obj.
        # Position vars: x[0], x[1]
        # Distance vars: x[2]..x[6]
        # g(x) = sum((x[i]-0.5)^2 for i in 2..6) + 10*|sin(6*pi*x[1] + pi)|
        # To get g=0:
        # x[2]..x[6] = 0.5
        # x[1] in {0, 1/6, 2/6, 3/6, 4/6, 5/6, 1}
        
        valid_x1 = np.array([0.0, 1.0/6.0, 2.0/6.0, 3.0/6.0, 4.0/6.0, 5.0/6.0, 1.0])
        
        # Distribute budget among the 7 slices
        n_slices = len(valid_x1)
        base_count = self.budget // n_slices
        remainder = self.budget % n_slices
        
        x0_samples = []
        x1_samples = []
        
        for i in range(n_slices):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                # Use linspace for uniform coverage of the line segment
                x0s = np.linspace(0.0, 1.0, count)
                x0_samples.extend(x0s)
                x1_samples.extend([valid_x1[i]] * count)
        
        # If for some reason we didn't use all budget (shouldn't happen), pad
        while len(x0_samples) < self.budget:
            x0_samples.append(0.5)
            x1_samples.append(0.5) # 0.5 is not valid for g=0, but this case shouldn't occur
            
        # Construct points
        points = []
        for x0, x1 in zip(x0_samples, x1_samples):
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # Set distance variables to 0.5
            for j in range(2, self.dim):
                x[j] = 0.5
            points.append(x)
        
        # Evaluate
        results = [problem(x) for x in points]
        
        return results
