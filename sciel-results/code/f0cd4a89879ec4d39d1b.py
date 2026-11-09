"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Non-stationary linear bandit with sliding-window and adaptive exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Window size for sliding window (in terms of effective samples)
        self.window = 100
        # Exploration parameter
        self.beta = 2.0
        # Initialize precision matrices A (inverse of covariance) and reward vectors b
        self.A = [np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        # Maintain the inverse of A for efficient UCB calculation
        self.V = [np.eye(self.d) for _ in range(self.n)]
        
        # Track the last time each arm was played for adaptive exploration
        self.last_played = [0] * self.n
        self.t = 0

    def select(self, context):
        scores = []
        for a in range(self.n):
            # Mean estimate: mu = V * b
            mu = self.V[a] @ self.b[a]
            # Predicted reward: mu^T * context
            pred_reward = np.dot(mu, context)
            
            # UCB Bonus: beta * sqrt(context^T V context)
            var = np.dot(context, self.V[a] @ context)
            if var < 0:
                var = 0.0
            bonus = self.beta * np.sqrt(var)
            
            # Adaptive exploration: increase bonus for arms not played recently
            time_since_played = self.t - self.last_played[a]
            if time_since_played > 10:
                bonus *= (1.0 + 0.1 * (time_since_played // 10))
            
            scores.append(pred_reward + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        self.last_played[arm] = self.t
        
        # Sliding window approach: decay old data
        # Use exponential decay with rate based on window size
        gamma = 0.99  # Effective window of ~100 samples
        
        # 1. Decay existing statistics
        self.V[arm] = self.V[arm] / gamma
        self.b[arm] = self.b[arm] * gamma
        
        # 2. Update with new observation using Woodbury Identity
        V_curr = self.V[arm]
        x = context
        v = V_curr @ x
        denom = 1.0 + np.dot(x, v)
        
        # V_new = V_curr - (v v^T) / denom
        self.V[arm] = V_curr - (np.outer(v, v) / denom)
        
        # Update b
        self.b[arm] = self.b[arm] + reward * x
# EVOLVE-BLOCK-END
