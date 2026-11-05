"""Non-stationary linear contextual bandit with dual-timescale Kalman filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Dual-timescale Kalman filter for non-stationary linear bandits.
    
    Models each arm's weight vector as a random walk:
      theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q_t)
      reward = theta_t^T x + noise,  noise ~ N(0, R)
    
    The process noise Q is adaptively scaled by prediction error to
    balance fast tracking of drift against stability.
    
    State: (theta, drift) where drift = theta_t - theta_{t-1}
    This allows the filter to separate fast reward estimation from slow drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 112.0

        # Kalman filter state per arm:
        # State vector: [theta (d), drift (d)] -> total 2d
        # theta: current weight estimate
        # drift: estimated change in theta per step
        
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.drift = [np.zeros(dim) for _ in range(n_arms)]
        
        # Covariance matrices (2d x 2d)
        # P = [[P_theta, P_theta_drift], [P_drift_theta, P_drift]]
        P_init = np.eye(2 * dim) * 10.0  # high initial uncertainty
        self.P = [P_init.copy() for _ in range(n_arms)]
        
        # Observation noise variance
        self.R = 1.0
        
        # Base process noise for theta (random walk variance per step)
        self.base_Q_theta = 0.015
        # Base process noise for drift (drift changes slowly)
        self.base_Q_drift = 0.001
        
        # Exploration coefficient for UCB
        self.C = 1.4
        self.C_min = 0.3
        self.C_max = 4.5
        
        # Counts and error tracking
        self.counts = [0] * n_arms
        self.pred_errors = [0.0] * n_arms
        self.error_counts = [0] * n_arms
        self.t = 0
        
        # EMA of squared innovation for adaptive process noise
        self.innovation_ema = [0.0] * n_arms
        self.ema_alpha = 0.15  # fast adaptation rate
        self.ema_beta = 0.3    # tanh steepness

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 2.0
        time_decay = max(0.35, 1.0 - 0.18 * (self.t / 500.0))
        return self.C * time_decay

    def _predict(self, arm):
        """Kalman predict step: propagate state and covariance forward."""
        d = self.d
        # State transition: theta_new = theta + drift, drift_new = drift
        # F = [[I, I], [0, I]]
        
        # Update state
        self.theta[arm] = self.theta[arm] + self.drift[arm]
        # drift stays the same in prediction (it's the expected change)
        
        # Adaptive process noise using innovation EMA with tanh response
        ema = self.innovation_ema[arm]
        # tanh-based scaling: aggressive increase when innovation is large
        tanh_val = np.tanh(self.ema_beta * np.sqrt(max(ema, 0.0)))
        # Scale ranges from ~1.0 (low error) to ~2.0+ (high error)
        scale = 1.0 + 1.0 * tanh_val
        
        # Process noise covariance Q (2d x 2d)
        # Q = [[Q_theta + Q_drift, Q_drift], [Q_drift, Q_drift]]
        Q_theta = self.base_Q_theta * scale
        Q_drift = self.base_Q_drift * (1.0 + 0.5 * tanh_val)
        
        # P_new = F P F^T + Q
        # F = [[I, I], [0, I]]
        # F P F^T:
        # Let P = [[P11, P12], [P21, P22]]
        # F P = [[P11+P12, P12+P22], [P21, P22]]
        # F P F^T = [[P11+P12+P21+P22, P12+P22], [P21+P22, P22]]
        
        P = self.P[arm]
        P11 = P[:d, :d]
        P12 = P[:d, d:]
        P21 = P[d:, :d]
        P22 = P[d:, d:]
        
        # F P F^T
        P11_new = P11 + P12 + P21 + P22
        P12_new = P12 + P22
        P21_new = P21 + P22
        P22_new = P22
        
        # Add process noise
        Q_mat = np.zeros((2*d, 2*d))
        Q_mat[:d, :d] = (Q_theta + Q_drift) * np.eye(d)
        Q_mat[:d, d:] = Q_drift * np.eye(d)
        Q_mat[d:, :d] = Q_drift * np.eye(d)
        Q_mat[d:, d:] = Q_drift * np.eye(d)
        
        P_new = np.zeros((2*d, 2*d))
        P_new[:d, :d] = P11_new + Q_mat[:d, :d]
        P_new[:d, d:] = P12_new + Q_mat[:d, d:]
        P_new[d:, :d] = P21_new + Q_mat[d:, :d]
        P_new[d:, d:] = P22_new + Q_mat[d:, d:]
        
        # Ensure symmetry
        P_new = (P_new + P_new.T) / 2.0
        
        self.P[arm] = P_new

    def _update(self, arm, context, reward):
        """Kalman update step: incorporate observation."""
        d = self.d
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Observation model: y = H state + noise
        # H = [I, 0] (observe theta only)
        H_theta = np.eye(d)
        H_drift = np.zeros((d, d))
        
        # Predicted observation
        pred = self.theta[arm] @ context
        
        # Innovation
        innovation = reward - pred
        
        # Innovation covariance: S = H P H^T + R
        P = self.P[arm]
        P11 = P[:d, :d]
        S = context @ P11 @ context + self.R
        
        if S < 1e-12:
            S = 1e-12
        
        # Kalman gain: K = P H^T S^{-1}
        # K is (2d x 1)
        PH = np.zeros((2*d, d))
        PH[:d, :] = P11
        PH[d:, :] = P[d:, :d]  # P21
        
        K = (PH @ context) / S  # (2d,)
        
        # Update state
        self.theta[arm] = self.theta[arm] + K[:d] * innovation
        self.drift[arm] = self.drift[arm] + K[d:] * innovation
        
        # Update covariance: P_new = (I - K H) P
        # I - K H: (2d x 2d)
        KH = np.zeros((2*d, 2*d))
        KH[:d, :d] = np.outer(K[:d], context)
        KH[d:, :d] = np.outer(K[d:], context)
        
        I_KH = np.eye(2*d) - KH
        self.P[arm] = I_KH @ P
        
        # Ensure symmetry and positive semi-definite
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Deadline guard: greedy
            scores = []
            for a in range(self.n):
                mu = self.theta[a] @ context
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # UCB bonus from theta uncertainty
            c_val = self._adaptive_C(a)
            P11 = self.P[a][:self.d, :self.d]
            var = context @ P11 @ context
            bonus = c_val * np.sqrt(max(0.0, var))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        self.counts[arm] += 1

        # Track prediction error for adaptive process noise
        pred = self.theta[arm] @ context
        err = reward - pred
        self.pred_errors[arm] += err * err
        self.error_counts[arm] += 1
        
        # Update innovation EMA (fast adaptation)
        self.innovation_ema[arm] = (1.0 - self.ema_alpha) * self.innovation_ema[arm] + self.ema_alpha * (err * err)

        # Kalman predict then update
        self._predict(arm)
        self._update(arm, context, reward)

        # Periodic numerical stabilization
        if self.t % 30 == 0:
            for a in range(self.n):
                P = self.P[a]
                eigvals, eigvecs = np.linalg.eigh(P)
                eigvals = np.clip(eigvals, 1e-8, 50.0)
                self.P[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Decay innovation EMA slowly to allow baseline reset
        if self.t % 100 == 0:
            for a in range(self.n):
                self.innovation_ema[a] *= 0.5
# EVOLVE-BLOCK-END
