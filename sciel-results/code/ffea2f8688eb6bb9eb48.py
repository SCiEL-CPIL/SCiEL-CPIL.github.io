"""Per-Arm Linear Kalman Filter with Adaptive Observation Noise."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear contextual bandit policy using a per-arm
    Linear Kalman Filter with adaptive observation noise variance.
    
    Instead of using a forgetting factor on the information matrix (which
    aggressively inflates process noise), we model the arm weights as a
    random walk (process noise Q) and adapt the observation noise R based
    on the magnitude of the standardized innovation. Large innovations
    suggest regime shifts or outliers, so we increase R to down-weight
    that observation, preserving estimator integrity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Initialize per-arm Kalman filter states
        # P: covariance matrix (d x d)
        # theta: estimated weight vector (d,)
        # We initialize P to a large identity to indicate high uncertainty
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Process noise covariance Q (random walk model for non-stationarity)
        # This allows the weights to drift over time.
        self.Q = 0.05 * np.eye(dim)

        # Base observation noise variance
        self.R_base = 1.0

        # Exploration bonus scaling
        self.alpha = 1.0

        # Track innovation history for adaptive R
        self.innovation_history = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, context):
        """
        Select arm using UCB-style exploration based on Kalman filter uncertainty.
        """
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            # Predicted reward
            mu = self.theta[a] @ context

            # Variance of predicted reward: context^T P context
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 0.0))

            # UCB score
            scores[a] = mu + self.alpha * std

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the Kalman filter for the selected arm.
        """
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        a = arm

        # --- Prediction Step ---
        # For a random walk model: theta_{k|k-1} = theta_{k-1|k-1}
        # P_{k|k-1} = P_{k-1|k-1} + Q
        self.P[a] = self.P[a] + self.Q

        # --- Update Step ---
        # Observation: y = theta^T x + noise
        # Innovation: nu = y - x^T theta
        predicted_reward = self.theta[a] @ context
        nu = reward - predicted_reward

        # Observation variance: S = x^T P x + R
        # Adaptive R: if |nu| is large relative to expected std, increase R
        Pxx = context @ self.P[a] @ context
        S_base = Pxx + self.R_base
        std_S_base = np.sqrt(max(S_base, 1e-12))

        # Standardized innovation
        if std_S_base > 1e-12:
            z = abs(nu) / std_S_base
        else:
            z = 0.0

        # Adaptive R: increase R when innovation is large
        # Use a smooth function: R_adaptive = R_base * (1 + z^2)
        # This down-weights large innovations
        R_adaptive = self.R_base * (1.0 + z ** 2)

        S = Pxx + R_adaptive

        # Kalman gain
        if S > 1e-12:
            K = self.P[a] @ context / S
        else:
            K = np.zeros(self.d)

        # Update theta
        self.theta[a] = self.theta[a] + K * nu

        # Update P (Joseph form for numerical stability)
        I_Kx = np.eye(self.d) - np.outer(K, context)
        self.P[a] = I_Kx @ self.P[a] @ I_Kx.T + R_adaptive * np.outer(K, K)

        # Ensure P is symmetric positive semi-definite
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
# EVOLVE-BLOCK-END
