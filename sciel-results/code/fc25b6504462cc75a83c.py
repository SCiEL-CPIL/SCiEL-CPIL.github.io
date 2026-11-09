"""
Non-stationary linear contextual bandit using Exponential Discounting RLS with UCB.

Strategy:
1. Exponential Discounting: Instead of a hard sliding window, we use an exponential
   discount factor (gamma) to weight recent observations more heavily. This provides
   smooth adaptation to non-stationarity.
2. Recursive Least Squares (RLS): We maintain a sufficient statistic (P matrix, theta vector)
   for each arm. Updates are performed in O(d^2) time using the matrix inversion lemma,
   which is significantly faster than recomputing from a window or inverting matrices
   from scratch every step.
3. UCB (Upper Confidence Bound): We use the standard LinUCB formula:
   score = theta @ x + alpha * sqrt(x @ P @ x)
   This provides a principled exploration bonus based on the model's uncertainty.
4. Deadline Guard: Ensures completion within the 150s limit.
"""

import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Exponential discounting factor.
        # Closer to 1 means slower forgetting (more stable), closer to 0 means faster forgetting (more reactive).
        # 0.95 is a common choice for non-stationary bandits.
        self.gamma = 0.95
        
        # Regularization parameter for RLS.
        # Prevents singularity in P.
        self.lambda_reg = 1.0
        
        # Exploration parameter for UCB.
        # Higher alpha means more exploration.
        # In non-stationary settings, we might want slightly higher exploration
        # to detect changes, but standard values often work well with good discounting.
        self.alpha = 1.5
        
        # Initialize P (inverse of A) and theta (weights) for each arm.
        # P = I / lambda, theta = 0
        self.P = [np.eye(dim) / self.lambda_reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            ctx = np.asarray(context, dtype=float)
            for a in range(self.n):
                score = self.theta[a] @ ctx
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Mean reward estimate
            mean_reward = self.theta[a] @ context
            
            # Uncertainty term: sqrt(x^T P x)
            # P is positive definite, so this is real and non-negative.
            # Compute x^T P x efficiently
            Px = self.P[a] @ context
            variance = context @ Px
            # Clamp to 0 to avoid numerical issues leading to negative variance
            variance = max(0.0, variance)
            std_dev = np.sqrt(variance)
            
            # UCB score
            score = mean_reward + self.alpha * std_dev
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # RLS Update with Exponential Discounting
        # The update equations for discounted RLS are:
        # P_t = (1/gamma) * (P_{t-1} - (P_{t-1} x x^T P_{t-1}) / (gamma + x^T P_{t-1} x))
        # theta_t = theta_{t-1} + P_t (y - x^T theta_{t-1}) x
        # Note: Some formulations use lambda in the denominator or adjust P differently.
        # Let's use the standard recursive form derived from minimizing:
        # sum_{i=1}^t gamma^{t-i} (y_i - x_i^T theta)^2 + lambda ||theta||^2
        
        # Let P_prev = P[arm]
        # Let x = context
        # Let y = reward
        
        P_prev = self.P[arm]
        theta_prev = self.theta[arm]
        
        # Compute denominator: gamma + x^T P_prev x
        Px = P_prev @ context
        denom = self.gamma + context @ Px
        
        # Compute P update
        # P_new = (1/gamma) * (P_prev - (P_prev x x^T P_prev) / denom)
        # To avoid full matrix multiplication P_prev @ x @ x^T @ P_prev, we use rank-1 update structure
        # term = (P_prev @ context) @ (context @ P_prev) / denom  =>  outer(Px, Px) / denom
        # But note the 1/gamma factor outside.
        
        # Optimization: 
        # P_new = (1/gamma) * P_prev - (1/gamma) * (outer(Px, Px) / denom)
        
        # However, standard RLS update is often written as:
        # K = P x / (lambda + x^T P x)  (Gain)
        # P = (1/gamma) * (P - K x^T P)
        # theta = theta + K (y - x^T theta)
        
        # Let's stick to the explicit matrix update for clarity and stability, 
        # as d is small (<=10), O(d^2) or O(d^3) is fine, but O(d^2) is better.
        
        # Using the form:
        # P_new = (1/gamma) * (P_prev - (P_prev @ context) @ (context @ P_prev) / (self.gamma + context @ (P_prev @ context)))
        
        # We already have Px = P_prev @ context
        # denom = self.gamma + context @ Px
        
        # Correction term: (Px @ context.T) is outer product, but we need (P_prev x) (x^T P_prev)
        # Which is outer(Px, Px).
        
        correction = np.outer(Px, Px) / denom
        
        # Apply discount and update P
        # P_new = (1/gamma) * P_prev - (1/gamma) * correction
        # This can be numerically unstable if gamma is small.
        # Alternative standard form:
        # P_new = (1/gamma) * (P_prev - (P_prev @ context) @ (context @ P_prev) / (self.gamma + context @ (P_prev @ context)))
        
        # Let's use the Gain form which is often more stable:
        # K = P_prev @ context / denom
        # P_new = (1/gamma) * (P_prev - np.outer(K, context) @ P_prev)
        # Wait, the matrix inversion lemma for (A + uv^T)^-1 is A^-1 - A^-1 u v^T A^-1 / (1 + v^T A^-1 u).
        # Here we are inverting sum gamma^i x_i x_i^T + lambda I.
        # Let A_t = gamma A_{t-1} + x x^T + lambda (1-gamma) I ? No, usually lambda is just initial.
        
        # Let's use the widely cited Discounted RLS update:
        # P_t = (1/gamma) * (P_{t-1} - (P_{t-1} x x^T P_{t-1}) / (gamma + x^T P_{t-1} x))
        # theta_t = theta_{t-1} + P_t (y - x^T theta_{t-1}) x
        
        # This is O(d^2) for the outer product and matrix subtraction.
        
        # Calculate P_new
        # To avoid allocating a new matrix for the correction, we can do:
        # P_new = P_prev / self.gamma
        # P_new -= (P_prev @ context) @ (context @ P_prev) / (self.gamma * denom)
        # Note: (P_prev @ context) @ (context @ P_prev) is outer(Px, Px).
        
        # Let's compute:
        P_new = P_prev / self.gamma
        # Subtract correction
        # correction_matrix = np.outer(Px, Px) / (self.gamma * denom)
        # P_new -= correction_matrix
        
        # For efficiency, we can update in place if we are careful, but P_prev is referenced.
        # Since d is small, creating a new array is fine and safer.
        
        # Actually, let's use the Gain vector approach which is cleaner:
        # K = P_prev @ context / denom
        # P_new = (1/gamma) * (P_prev - np.outer(K, context) @ P_prev)
        # This is also O(d^2).
        
        # Let's verify the Gain formula equivalence:
        # P_prev - K x^T P_prev = P_prev - (P_prev x / (gamma + x^T P_prev x)) x^T P_prev
        # = P_prev - (P_prev x x^T P_prev) / (gamma + x^T P_prev x)
        # Yes, it matches.
        
        K = Px / denom
        # P_new = (1/gamma) * (P_prev - np.outer(K, context) @ P_prev)
        # Note: np.outer(K, context) @ P_prev is (d x d) @ (d x d) which is O(d^3).
        # The previous form: P_prev - np.outer(Px, Px)/denom is O(d^2) for outer + O(d^2) for subtraction.
        # So the direct matrix subtraction form is faster.
        
        # Let's revert to the O(d^2) update:
        # P_new = P_prev / self.gamma
        # P_new -= np.outer(Px, Px) / (self.gamma * denom)
        
        # Let's implement this carefully.
        P_new = P_prev / self.gamma
        # Compute outer product
        # Px is shape (d,). outer(Px, Px) is (d, d).
        # We can use np.outer(Px, Px)
        corr = np.outer(Px, Px)
        P_new -= corr / (self.gamma * denom)
        
        # Ensure symmetry (numerical drift)
        P_new = (P_new + P_new.T) / 2.0
        
        # Update theta
        # theta_new = theta_prev + P_new * (reward - context @ theta_prev)
        # Note: P_new here is the updated P.
        # Some formulations use the old P in the theta update?
        # Standard RLS: theta_t = theta_{t-1} + K (y - x^T theta_{t-1})
        # Where K = P_t x.
        # So theta_new = theta_prev + (P_new @ context) * (reward - context @ theta_prev)
        
        error = reward - context @ theta_prev
        # K_new = P_new @ context
        # But we can reuse Px? No, Px was P_prev @ context.
        # We need P_new @ context.
        # Since P_new is derived from P_prev, we can compute it.
        # Or we can use the relationship:
        # P_new x = (1/gamma) P_prev x - (1/gamma) (Px Px^T / denom) x
        # = (1/gamma) Px - (1/gamma) (Px (Px^T x) / denom)
        # Px^T x = x^T Px = denom - gamma
        # So P_new x = (1/gamma) Px - (1/gamma) Px (denom - gamma) / denom
        # = (1/gamma) Px [ 1 - (denom - gamma)/denom ]
        # = (1/gamma) Px [ gamma/denom ]
        # = Px / denom
        # = K (the gain vector computed earlier!)
        
        # So the update for theta is:
        # theta_new = theta_prev + K * error
        
        theta_new = theta_prev + K * error
        
        self.P[arm] = P_new
        self.theta[arm] = theta_new
        
        self.t += 1
