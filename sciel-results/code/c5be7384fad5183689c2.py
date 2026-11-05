# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a Kalman filter-style
    recursive least squares (RLS) estimator with adaptive observation noise.
    
    For each arm, we maintain a state vector theta and its covariance P.
    The update step uses a Kalman filter formulation where:
    - Process noise Q models gradual drift in the weights.
    - Observation noise R is adapted to down-weight large innovations (regime shifts).
    
    This approach is more stable than fixed forgetting factors because it
    explicitly models the uncertainty in the weights and adapts the trust
    in new observations based on their magnitude relative to expected noise.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Initialize per-arm Kalman filter states
        # theta: current weight estimate
        # P: covariance of the weight estimate
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # Hyperparameters
        self.Q = 0.01 * np.eye(dim)  # Process noise (drift)
        self.R_base = 1.0            # Base observation noise
        self.R_min = 0.1             # Minimum observation noise
        self.R_max = 10.0            # Maximum observation noise
        self.R_alpha = 0.95          # Adaptation rate for R
        self.exploration = 1.0       # UCB coefficient
        self.ridge = 1e-6            # Small ridge for numerical stability

        # Track R per arm for adaptive noise
        self.R = [self.R_base for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Fallback to a simple heuristic if we're about to timeout
            # In practice, this shouldn't happen often, but it's a safety net
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted reward
            mu = self.theta[a] @ context
            # Uncertainty (variance of the prediction)
            var = context @ self.P[a] @ context
            # UCB score
            score = mu + self.exploration * np.sqrt(max(var, 0.0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Kalman filter update for arm
        P = self.P[arm]
        theta = self.theta[arm]

        # Innovation
        innovation = reward - theta @ context

        # Adaptive observation noise: down-weight large innovations
        # If the innovation is large, increase R to trust the prior more
        R = self.R[arm]
        # Update R based on innovation magnitude
        # Use a robust adaptation: R_new = alpha * R_old + (1-alpha) * max(R_min, innovation^2 / d)
        # But we want to down-weight outliers, so if innovation is large, R should be large
        # Actually, we want to *reduce* the weight of the observation, which means *increasing* R
        # Let's use: R_adapted = R * (1 + beta * |innovation|)
        # Or simpler: if |innovation| > threshold, increase R
        # Let's use the measured strategy: adaptive R down-weights large innovations
        # We'll update R to be proportional to the innovation magnitude, but bounded
        R_new = self.R_base * (1.0 + 0.5 * abs(innovation))
        R_new = min(R_new, self.R_max)
        R_new = max(R_new, self.R_min)
        self.R[arm] = R_new

        # Kalman gain
        S = context @ P @ context + R_new
        K = (P @ context) / S

        # Update theta
        theta_new = theta + K * innovation
        # Update P (Joseph form for numerical stability)
        P_new = (P - np.outer(K, context @ P)) + self.Q
        # Ensure symmetry
        P_new = (P_new + P_new.T) / 2.0
        # Add ridge for stability
        P_new += self.ridge * np.eye(self.d)

        self.theta[arm] = theta_new
        self.P[arm] = P_new
# EVOLVE-BLOCK-END
