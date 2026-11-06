import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global model state: A (d x d), b (d)
        # Regularized with identity to ensure invertibility
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        
        # Arm-specific biases
        # beta_a is the scalar bias for arm a
        # We can also track count for each arm to weight the bias update if needed, 
        # but simple RLS on bias is sufficient.
        self.beta = np.zeros(n_arms)
        
        # Hyperparameters
        # lambda for global model: balance between forgetting and stability
        # Since non-stationary, we want a moderate forgetting factor
        self.lambda_global = 0.95
        
        # lambda for bias: biases might change slower or faster? 
        # Let's use a slightly higher lambda (slower forgetting) for biases 
        # as arm-specific offsets might be more stable than the context-dependent part
        self.lambda_bias = 0.98
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Timing
        self.start_time = time.time()
        self.deadline = 110.0
        
        # Error tracking for adaptive alpha
        self.recent_errors = []
        self.max_error_window = 50

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def select(self, context):
        if not self._check_deadline():
            # Fallback: pick arm with highest bias (exploit known best offset)
            return int(np.argmax(self.beta))

        context = np.asarray(context)
        
        # Compute global theta
        # A is always invertible because we start with I and add rank-1 updates with lambda < 1
        # However, to be safe against numerical issues, we use solve or pinv if needed.
        # For speed, we assume A is well-conditioned.
        try:
            inv_A = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            inv_A = np.eye(self.d)
            
        theta = inv_A @ self.b
        
        # Prediction for each arm: theta @ x + beta_a
        # Variance for each arm: alpha * sqrt(x @ inv_A @ x)
        # Note: The variance term is the same for all arms because it comes from the global model uncertainty.
        # The bias beta_a is treated as a deterministic offset in this simple formulation.
        # To account for bias uncertainty, we could add a term, but let's keep it simple first.
        # If we want to be more rigorous, we'd need to track bias variance.
        # For now, the exploration bonus is driven by the global model uncertainty.
        
        x_inv_A_x = context @ inv_A @ context
        bonus = self.alpha * np.sqrt(max(0.0, x_inv_A_x))
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            mu = theta @ context + self.beta[a]
            scores[a] = mu + bonus
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return
            
        context = np.asarray(context)
        
        # 1. Compute prediction error before update
        try:
            inv_A_old = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            inv_A_old = np.eye(self.d)
            
        theta_old = inv_A_old @ self.b
        pred = theta_old @ context + self.beta[arm]
        residual = reward - pred
        
        # Track error for adaptive alpha
        self.recent_errors.append(residual)
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
            
        # 2. Update global model (RLS with forgetting)
        # A_new = lambda * A_old + x x^T
        # b_new = lambda * b_old + r x
        self.A = self.lambda_global * self.A + np.outer(context, context)
        self.b = self.lambda_global * self.b + reward * context
        
        # 3. Update arm-specific bias
        # Treat the residual as the signal to update the bias?
        # Or treat the reward as the signal?
        # If we use reward, we are fitting beta to the raw reward, which includes the context effect.
        # That's bad because theta already captures the context effect.
        # We should fit beta to the residual of the global model?
        # Let's try updating beta based on the residual.
        # beta_new = lambda_bias * beta_old + residual
        # This makes beta track the average prediction error of the global model for that arm.
        self.beta[arm] = self.lambda_bias * self.beta[arm] + residual
        
        # 4. Adapt alpha
        if len(self.recent_errors) >= 10:
            mse = np.mean(np.array(self.recent_errors)**2)
            # Map MSE to alpha
            # Low MSE -> Low Alpha (exploit)
            # High MSE -> High Alpha (explore)
            # Scale: assume typical MSE is around 1.0
            target_alpha = 0.5 + 1.5 * (mse / (mse + 0.5))
            self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
            self.alpha = max(0.2, min(2.0, self.alpha))

