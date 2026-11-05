"""Non-stationary linear contextual bandit with RLS and adaptive forgetting factor."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Recursive Least Squares (RLS)
    with an adaptive forgetting factor.
    
    Model: r_t = theta_t^T x_t + noise
    Non-stationarity is handled by exponentially weighting past observations.
    The forgetting factor gamma controls the memory length.
    Adaptive gamma: decreases (shorter memory) when prediction errors are high,
    increases (longer memory) when errors are low.
    
    Selection: UCB based on RLS estimate + alpha * posterior std.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm RLS state
        # P_a is the inverse of the weighted information matrix A_a
        # Initialized to lambda * I to represent high initial uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor parameters
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        self.gamma = 0.95  # Initial gamma

        # UCB exploration coefficient
        self.alpha = 1.0

        # Adaptive gamma tracking
        # We track the average squared innovation to normalize current error
        self.innovation_sq_sum = 0.0
        self.innovation_count = 0
        self.avg_inn_sq = 1.0  # Initial guess for variance of noise
        self.inn_window = 50   # Effective window for adaptation

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def _update_gamma(self, innovation):
        """
        Adapt the forgetting factor based on the magnitude of the innovation.
        Large innovation -> likely drift -> reduce gamma (forget faster).
        Small innovation -> stable -> increase gamma (remember more).
        """
        # Update running average of squared innovations
        # Use exponential moving average for efficiency
        self.avg_inn_sq = 0.95 * self.avg_inn_sq + 0.05 * (innovation * innovation)
        
        # Normalize current innovation
        # If avg_inn_sq is very small, avoid division by zero
        norm_inn = abs(innovation) / (np.sqrt(self.avg_inn_sq) + 1e-6)

        # Adjust gamma
        # If norm_inn > 1.0, error is larger than average -> drift suspected
        if norm_inn > 1.5:
            # Increase drift response: decrease gamma
            self.gamma = max(self.gamma_min, self.gamma * 0.95)
        elif norm_inn < 0.5:
            # Stable period: increase gamma to reduce variance
            self.gamma = min(self.gamma_max, self.gamma * 1.02)
        else:
            # Neutral zone: slight decay towards a middle value? 
            # Or just keep current. Let's gently pull towards 0.95
            self.gamma = 0.99 * self.gamma + 0.01 * 0.95

    def select(self, context):
        """Select arm using UCB with RLS posterior."""
        if self._check_deadline():
            # Fallback: greedy on mean
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        x = context

        ucb_scores = []
        for a in range(self.n):
            mean = self.theta[a] @ x
            # Posterior variance: x^T P_a x
            var = x @ self.P[a] @ x
            std = np.sqrt(max(var, 0.0))
            ucb = mean + self.alpha * std
            ucb_scores.append(ucb)

        return int(np.argmax(ucb_scores))

    def update(self, context, arm, reward):
        """
        RLS update for the selected arm.
        Updates P and theta.
        """
        context = np.asarray(context, dtype=np.float64)
        x = context
        r = float(reward)
        
        # 1. Calculate innovation (prediction error)
        pred = self.theta[arm] @ x
        innovation = r - pred
        
        # 2. Adapt forgetting factor
        self._update_gamma(innovation)
        
        # 3. RLS Update
        # Let P be the current inverse information matrix.
        # The update rule for P in RLS with forgetting factor gamma is:
        # P_new = (1/gamma) * (P_old - (P_old x x^T P_old) / (gamma + x^T P_old x))
        
        P = self.P[arm]
        
        # Compute denominator
        # x^T P x
        P_x = P @ x  # shape (d,)
        denom = self.gamma + x @ P_x
        
        # Compute gain
        # K = P_x / denom
        # But we need to update P and theta.
        
        # Update P
        # P_new = (1/gamma) * (P - (P_x P_x^T) / denom)
        # Note: P_x P_x^T is a rank-1 update.
        outer = np.outer(P_x, P_x)
        P_new = (P - outer / denom) / self.gamma
        
        # Update theta
        # theta_new = theta_old + K * innovation
        # Where K = P_new x ? No, standard RLS gain is K = P_old x / (gamma + x^T P_old x)
        # Let's verify.
        # theta_new = A_new^{-1} b_new
        # A_new = gamma A_old + x x^T
        # b_new = gamma b_old + x r
        # theta_new = (gamma A_old + x x^T)^{-1} (gamma b_old + x r)
        # Using Woodbury or matrix inversion lemma:
        # (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma A_old, u = x, v = x.
        # (gamma A_old + x x^T)^{-1} = (1/gamma) A_old^{-1} - (1/gamma) A_old^{-1} x x^T A_old^{-1} / (1 + (1/gamma) x^T A_old^{-1} x)
        # = (1/gamma) [ P_old - P_old x x^T P_old / (gamma + x^T P_old x) ]
        # This matches the P_new formula above.
        
        # Now for theta:
        # theta_new = P_new (gamma theta_old + x r)  ? No, theta_old = P_old b_old.
        # b_new = gamma b_old + x r
        # theta_new = P_new b_new = P_new (gamma b_old + x r)
        # = P_new (gamma A_old theta_old + x r)
        # = P_new (gamma (A_old theta_old) + x r)
        # Since A_old = P_old^{-1}, this is getting complex.
        
        # Simpler RLS theta update:
        # theta_new = theta_old + K * innovation
        # Where K = P_old x / (gamma + x^T P_old x)
        
        K = P_x / denom
        self.theta[arm] = self.theta[arm] + K * innovation
        self.P[arm] = P_new
        
        # Ensure P is symmetric (numerical stability)
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
# EVOLVE-BLOCK-END
