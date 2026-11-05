"""Non-stationary linear contextual bandit with adaptive Kalman forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Kalman Filter for non-stationary linear bandits.
    Uses innovation-based process noise to dynamically adjust memory length.
    Includes a deadline guard to ensure termination within wall-clock limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit

        # Per-arm Kalman state
        # P: Covariance matrix (d x d)
        # theta: Weight vector (d,)
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive process noise parameters
        # Q_base: Baseline process noise (small)
        # Q_max: Maximum process noise (large, for rapid adaptation)
        # innovation_threshold: Threshold for innovation magnitude to trigger adaptation
        # decay: How quickly Q returns to base after innovation drops
        self.Q_base = 0.001
        self.Q_max = 5.0
        self.innovation_threshold = 2.0  # Standard deviations
        self.decay = 0.95

        # Track innovation per arm to adapt Q
        self.current_Q = [self.Q_base for _ in range(n_arms)]

        # Exploration bonus coefficient
        self.alpha = 1.0

        # Counters
        self.t = 0

    def select(self, context):
        # Deadline guard: if we are too close to the limit, fall back to greedy
        # based on current estimates to avoid heavy computation or hanging.
        if time.time() > self.deadline:
            # Greedy fallback
            scores = [float(self.theta[a] @ context) for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Prediction: theta_pred = theta, P_pred = P + Q
            # We don't explicitly store pred, we just compute uncertainty for UCB
            # Uncertainty: sqrt(context^T P context)
            # Note: In Kalman filter, the posterior covariance is P.
            # The "prediction" covariance for the next step would be P + Q.
            # For UCB, we typically use the posterior uncertainty.
            
            p_val = self.P[a]
            theta_val = self.theta[a]
            
            # Compute mean reward estimate
            mean = float(theta_val @ context)
            
            # Compute uncertainty
            # context^T P context
            tmp = p_val @ context
            variance = float(context @ tmp)
            if variance < 0:
                variance = 0.0
            std = np.sqrt(variance)
            
            # UCB score
            score = mean + self.alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard: skip update if too close to limit to save time
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        a = arm

        # 1. Innovation (Prediction Error)
        # Predicted reward: theta @ context
        pred = float(self.theta[a] @ context)
        innovation = reward - pred

        # 2. Adaptive Process Noise
        # If innovation is large, the model is wrong -> increase Q for next step
        # If innovation is small, model is good -> decrease Q towards base
        abs_innov = abs(innovation)
        if abs_innov > self.innovation_threshold:
            # Increase Q proportionally to error magnitude, capped
            new_Q = min(self.Q_max, self.current_Q[a] + abs_innov * 0.1)
        else:
            # Decay Q towards base
            new_Q = self.Q_base + (self.current_Q[a] - self.Q_base) * self.decay

        # 3. Kalman Update
        # Current covariance P
        P = self.P[a]
        theta = self.theta[a]

        # Kalman Gain: K = P H^T (H P H^T + R)^-1
        # Here H = context (1xd), so H^T is dx1
        # H P H^T is scalar: context^T P context
        S = float(context @ (P @ context))  # Innovation variance (without measurement noise R, or assume R=1)
        # Assume measurement noise R = 1.0 for stability, or tune.
        # Let's use R = 1.0 to prevent overfitting to single noisy points.
        R = 1.0
        S += R

        if S < 1e-9:
            S = 1e-9

        # K = P context / S
        K = (P @ context) / S  # (d,)

        # Update theta
        self.theta[a] = theta + K * innovation

        # Update P (Joseph form for numerical stability)
        # P_new = (I - K H) P (I - K H)^T + K R K^T
        # Simplified: P_new = P - K (H P)
        # H P = context^T P (1xd)
        # K (H P) = (d x 1) (1 x d) = d x d
        # Let's use the standard form: P = (I - K context^T) P
        # I - K context^T
        I = np.eye(self.d)
        IK = I - np.outer(K, context)
        self.P[a] = IK @ P @ IK.T + np.outer(K, K) * R

        # Ensure symmetry and positive semi-definiteness
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # 4. Predict step (add process noise)
        # P = P + Q
        # We update current_Q for the NEXT prediction step
        # But we also need to add the CURRENT Q to P now?
        # In standard KF:
        # 1. Predict: P = P + Q
        # 2. Update: ...
        # Here we are doing Update first because we received data.
        # So we should add the Q that was active *before* this update to P?
        # Or, typically in online bandits, we treat the change as happening *after* the observation.
        # Let's add the current_Q to P after the update, representing the uncertainty introduced by the potential drift.
        
        # Actually, it's cleaner to:
        # 1. Use current_Q in the prediction step (which we effectively did by using P from previous step).
        # 2. Update P with innovation.
        # 3. Add new_Q (or current_Q?) to P for the next step.
        
        # Let's stick to: P_new = P_updated + Q_current
        # Then set current_Q = new_Q for the next iteration's prediction.
        
        self.P[a] += self.current_Q[a] * np.eye(self.d)
        
        # Update the Q for the next step
        self.current_Q[a] = new_Q

        self.t += 1
# EVOLVE-BLOCK-END
