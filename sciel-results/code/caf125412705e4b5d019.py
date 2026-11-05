"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive LinUCB with dynamic forgetting and exploration.
    
    Key differences from baseline:
    1. Adaptive forgetting factor: beta is adjusted based on recent reward variance.
       High variance -> lower beta (faster forgetting) to track changes.
       Low variance -> higher beta (slower forgetting) to reduce noise.
    2. Dynamic alpha: Exploration decays over time but can spike if uncertainty is high.
    3. Deadline guard: Ensures the program terminates safely within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initial parameters
        self.beta = 0.95  # Start with faster forgetting than 0.99 to be more agile
        self.alpha = 1.5  # Start with higher exploration
        
        # Tracking variables
        self.t = 0
        self.recent_rewards = []
        self.reward_window_size = 10
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0 # Safety margin (120s limit)
        
        # For adaptive beta calculation
        self.reward_mean = 0.0
        self.reward_var = 1.0

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback to a safe, fast decision if we are running out of time
            # In a real bandit setting, we just pick the best estimated arm quickly
            # Here we return the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            try:
                # Solve A * theta = b
                theta = np.linalg.solve(self.A[a], self.b[a])
                
                # Prediction
                mu = theta @ context
                
                # Confidence interval
                # var = context^T A^{-1} context
                inv_A_ctx = np.linalg.solve(self.A[a], context)
                var = context @ inv_A_ctx
                if var < 0:
                    var = 0.0
                uc = self.alpha * np.sqrt(var)
                
                scores.append(mu + uc)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Check deadline
        if time.time() > self.deadline:
            # If we are close to deadline, we might want to stop updating complex stats
            # but still update the model minimally to not crash
            self.A[arm] = self.beta * self.A[arm]
            self.b[arm] = self.beta * self.b[arm]
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context
            return

        # 1. Update adaptive forgetting factor based on reward variance
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window_size:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= 5:
            r_mean = np.mean(self.recent_rewards)
            r_var = np.var(self.recent_rewards)
            
            # Map variance to beta
            # If variance is high, we want to forget faster (lower beta)
            # If variance is low, we want to remember longer (higher beta)
            # Base beta range: [0.90, 0.99]
            # Normalize variance: high var -> 0, low var -> 1
            # Use a simple scaling. Assume rewards are in a reasonable range.
            # If var > 1.0, treat as high. If var < 0.1, treat as low.
            var_norm = min(1.0, max(0.0, (1.0 - r_var) / 0.9))
            
            # Target beta: 0.90 + 0.09 * var_norm
            target_beta = 0.90 + 0.09 * var_norm
            
            # Smoothly adjust beta to target
            self.beta = 0.9 * self.beta + 0.1 * target_beta

        # 2. Update dynamic alpha
        # Alpha should generally decrease as we learn, but increase if uncertainty is high
        # Simple decay: alpha = max(alpha_min, alpha * decay_factor)
        # But let's also scale by the inverse of the number of steps slightly
        # To avoid over-exploration late in the game
        self.alpha = max(0.5, self.alpha * 0.999) 
        
        # If the reward was an outlier, temporarily boost alpha?
        # For simplicity, we stick to the decay, but the adaptive beta helps track changes.

        # 3. Update model with forgetting
        self.A[arm] = self.beta * self.A[arm]
        self.b[arm] = self.beta * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
# EVOLVE-BLOCK-END
