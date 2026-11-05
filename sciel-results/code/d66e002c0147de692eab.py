import numpy as np
import time

class Policy:
    """
    Global Linear Bandit with Adaptive Forgetting Factor.
    Context is augmented with one-hot arm indicators.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_total = dim + n_arms
        
        # Single global model
        self.A = np.eye(self.d_total)
        self.b = np.zeros(self.d_total)
        
        # Adaptive parameters from knowledge
        self.lambda_val = 0.95
        self.lambda_min = 0.78
        self.lambda_max = 0.98
        self.beta = 1.2
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _augment(self, context, arm):
        # context is (d,), arm is int
        # Create (d+n,) vector
        z = np.zeros(self.d_total)
        z[:self.d] = context
        z[self.d + arm] = 1.0
        return z

    def select(self, context):
        if time.time() > self.deadline:
            return 0
            
        scores = []
        # Solve once for theta if possible, but we need inverse for variance per arm
        # A is same for all arms in global model, so inv(A) is same
        try:
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
        except:
            A_inv = np.eye(self.d_total)
            theta = np.zeros(self.d_total)
            
        for a in range(self.n):
            z = self._augment(context, a)
            pred = theta @ z
            var = z @ A_inv @ z
            score = pred + self.beta * np.sqrt(var)
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        self.t += 1
        z = self._augment(context, arm)
        
        # Calculate error for adaptivity
        try:
            theta = np.linalg.solve(self.A, self.b)
            pred = theta @ z
            error = reward - pred
            abs_error = abs(error)
            
            # Adaptive Lambda
            if abs_error > 2.0:
                self.lambda_val = max(self.lambda_min, self.lambda_val - 0.05)
            else:
                self.lambda_val = min(self.lambda_max, self.lambda_val + 0.002)
        except:
            pass

        # Update Global Model
        lam = self.lambda_val
        self.A = lam * self.A + np.outer(z, z)
        self.b = lam * self.b + reward * z
