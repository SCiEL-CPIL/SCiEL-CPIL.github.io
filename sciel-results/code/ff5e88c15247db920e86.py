"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with context-augmented features (one-hot arm + context).
    Uses a single shared parameter vector to capture common trends and arm-specific offsets.
    Includes forgetting factor for non-stationarity and adaptive exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Feature dimension: one-hot arm (n) + context (dim)
        self.feature_dim = n + dim
        # Initialize global linear model
        self.A = np.eye(self.feature_dim) * 1.0  # Regularization
        self.b = np.zeros(self.feature_dim)
        # Forgetting factor for non-stationary rewards
        self.forgetting = 0.99
        # Exploration parameter
        self.alpha = 1.0
        # Track cumulative reward for adaptive alpha
        self.total_reward = 0.0
        self.n_updates = 0

    def _make_features(self, arm, context):
        """Create augmented feature vector: one-hot arm + context."""
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate([one_hot, context])

    def select(self, context):
        """
        Select arm using UCB with global model.
        For each arm, compute predicted reward + exploration bonus.
        """
        scores = []
        for a in range(self.n):
            features = self._make_features(a, context)
            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(self.A, self.b)
            except np.linalg.LinAlgError:
                theta = self.b / (np.sum(self.A) + 1e-10)
            
            # Predicted reward
            mu = theta @ features
            
            # Exploration bonus: alpha * sqrt(features^T A^{-1} features)
            try:
                inv_A = np.linalg.inv(self.A)
                variance = features @ inv_A @ features
                bonus = self.alpha * np.sqrt(max(variance, 0))
            except np.linalg.LinAlgError:
                bonus = 0
            
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the global linear model with the observed reward.
        Uses forgetting factor to handle non-stationarity.
        """
        features = self._make_features(arm, context)
        
        # Apply forgetting
        self.A = self.forgetting * self.A
        self.b = self.forgetting * self.b
        
        # Update with new observation
        self.A += np.outer(features, features)
        self.b += reward * features
        
        # Track statistics for adaptive exploration
        self.total_reward += reward
        self.n_updates += 1
        
        # Adapt alpha: decrease exploration over time but keep some minimum
        # Use a simple decay that maintains reasonable exploration
        base_alpha = 1.0
        decay_factor = 1.0 / (1.0 + 0.001 * self.n_updates)
        self.alpha = base_alpha * decay_factor + 0.1
# EVOLVE-BLOCK-END
