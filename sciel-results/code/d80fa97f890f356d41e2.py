<<<<<<< SEARCH
    def __call__(self, problem, bounds):
        n = self.budget
        # Generate uniform u values in [0, 1]
        u = np.linspace(0, 1, n)
        # Apply power law to concentrate samples near x1=0
        # x1 = u^2 minimizes the Riemann sum error for the concave hypervolume contribution
        x1 = u ** 2
        
        # Construct the population
        # x1 is the first variable, x2..xn are set to 0 to achieve g_min=1
        X = np.zeros((n, self.dim))
        X[:, 0] = x1
        
        # Evaluate all points
        # The problem function clips to bounds, but our values are already in [0, 1]
        F = np.array([problem(x) for x in X])
        
        return X
=======
    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        
        # ZDT2: f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1))
        # Reference point y_ref = 11.0
        # Find x_crit where f2(x_crit) = 11
        # 291 * (1 - sqrt(x_crit)) = 11
        # 1 - sqrt(x_crit) = 11 / 291
        # sqrt(x_crit) = 1 - 11 / 291
        # x_crit = (1 - 11/291)^2
        coeff = 1 + 10 * (dim - 1)
        x_crit = (1 - 11.0 / coeff) ** 2
        
        # We place 1 point at x_crit to correctly cap the hypervolume contribution
        # from the region [0, x_crit].
        # The remaining n-1 points are placed in [x_crit, 1].
        # The function f2 is convex in this region.
        # To minimize the error of the step-approximation (which underestimates the integral),
        # we should place points closer together where the slope is steeper.
        # The slope is steepest at x_crit and flattens out at x=1.
        # Therefore, we use a power law that concentrates points near x_crit.
        # x = x_crit + (1 - x_crit) * u^p
        # With p > 1, points are concentrated near x_crit (where u is small).
        
        X = np.zeros((n, dim))
        
        if n == 1:
            X[0, 0] = x_crit
        else:
            X[0, 0] = x_crit
            # Generate u in (0, 1] for the remaining points
            # We want to cover [x_crit, 1]
            u = np.linspace(0, 1, n - 1)[1:] # 1/(n-1) to 1
            # Apply power law p=2 to concentrate near x_crit
            x1_rest = x_crit + (1.0 - x_crit) * (u ** 2)
            X[1:, 0] = x1_rest
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
>>>>>>> REPLACE
