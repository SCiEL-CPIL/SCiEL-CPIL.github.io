"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Bayesian Linear Bandit with Thompson Sampling and adaptive forgetting.
    Uses the matrix inversion lemma for efficient covariance updates and 
    adapts the forgetting factor based on prediction errors to handle non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        
        # Initialize inverse covariance matrices and posterior means
        # V = A^{-1}, m = A^{-1}b
        self.V = [np.eye(dim) / self.lam for _ in range(n_arms)]
        self.m = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting parameters
        self.beta_min = 0.95
        self.beta_max = 0.999
        self.current_beta = 0.99
        
        # Noise variance estimate
        self.sigma2 = 1.0
        
        # Residual tracking for adaptive beta
        self.recent_residuals = []
        self.residual_window = 20
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > 110:
            return 0

        scores = []
        for a in range(self.n):
            # Mean prediction
            mu = self.m[a] @ context
            
            # Sample theta from N(m, V)
            # V is symmetric positive definite
            try:
                # Use Cholesky for sampling
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d)
                theta_sample = mu + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (shouldn't happen with ridge)
                theta_sample = mu + np.random.randn(self.d) * 0.1

            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # 1. Adaptive Forgetting Factor Calculation
        # Calculate prediction error (residual)
        mu_pred = self.m[arm] @ context
        residual = reward - mu_pred
        
        # Track recent residuals to estimate variance and detect shifts
        self.recent_residuals.append(residual)
        if len(self.recent_residuals) > self.residual_window:
            self.recent_residuals.pop(0)
        
        if len(self.recent_residuals) > 5:
            # Estimate noise variance
            self.sigma2 = max(1e-6, np.var(self.recent_residuals) + 1e-6)
            
            # Calculate Mahalanobis distance of the residual
            # This measures how "surprising" the new data point is given the current model uncertainty
            try:
                # The variance of the prediction is context^T V context
                pred_var = context @ (self.V[arm] @ context)
                # Standardized residual
                if pred_var > 1e-8:
                    mahalanobis_dist = abs(residual) / np.sqrt(pred_var + self.sigma2)
                else:
                    mahalanobis_dist = abs(residual) / np.sqrt(self.sigma2)
            except Exception:
                mahalanobis_dist = 1.0

            # Adjust beta based on Mahalanobis distance
            # If the error is large (high Mahalanobis distance), we are likely in a regime shift
            # We want to forget old data quickly -> lower beta
            # If the error is small, we are in a stable regime -> higher beta
            # Map Mahalanobis distance to beta range [beta_min, beta_max]
            # A distance of 0 implies perfect prediction -> beta_max
            # A distance of 3+ (3 sigma) implies shift -> beta_min
            if mahalanobis_dist < 0.5:
                self.current_beta = self.beta_max
            elif mahalanobis_dist > 3.0:
                self.current_beta = self.beta_min
            else:
                # Linear interpolation between 0.5 and 3.0
                ratio = (3.0 - mahalanobis_dist) / (3.0 - 0.5)
                self.current_beta = self.beta_min + ratio * (self.beta_max - self.beta_min)
        else:
            self.current_beta = 0.99

        # 2. Update Inverse Covariance and Mean using Matrix Inversion Lemma
        # With forgetting:
        # A_new = beta * A_old + x x^T
        # V_new = (A_new)^{-1}
        
        # We maintain V = A^{-1} and m = A^{-1}b
        # Let's update V and m directly.
        
        # First, decay the existing V and m
        # Note: (beta * A)^{-1} = (1/beta) * A^{-1}
        # So V_decayed = (1/beta) * V
        # m_decayed = (1/beta) * m? No, m = A^{-1}b. If A -> beta A, and b -> beta b, then m stays same?
        # Let's re-derive.
        # We want to model: A_t = beta * A_{t-1} + x_t x_t^T
        #               b_t = beta * b_{t-1} + r_t x_t
        
        # We store V = A^{-1} and m = A^{-1}b.
        
        # Step 2a: Decay V and m
        # If A_t = beta A_{t-1}, then V_t = (1/beta) V_{t-1}.
        # b_t = beta b_{t-1}.
        # m_t = A_t^{-1} b_t = (1/beta) V_{t-1} * beta b_{t-1} = V_{t-1} b_{t-1} = m_{t-1}.
        # So, decay V by 1/beta, keep m unchanged.
        
        V_decayed = self.V[arm] / self.current_beta
        m_decayed = self.m[arm]
        
        # Step 2b: Update with new observation (x, r)
        # A_new = A_decayed + x x^T
        # V_new = (A_decayed + x x^T)^{-1}
        # Using Sherman-Morrison: (M + u v^T)^{-1} = M^{-1} - (M^{-1} u v^T M^{-1}) / (1 + v^T M^{-1} u)
        # Here M = A_decayed, u = x, v = x.
        # M^{-1} = V_decayed.
        
        Vx = V_decayed @ context
        denom = 1.0 + context @ Vx
        
        if denom < 1e-6:
            # Numerical safety
            V_new = V_decayed
            m_new = m_decayed
        else:
            V_new = V_decayed - np.outer(Vx, context) / denom
            # Update mean
            # m_new = A_new^{-1} b_new
            # b_new = b_decayed + r x
            # We don't store b, we store m.
            # m_new = (A_decayed + x x^T)^{-1} (b_decayed + r x)
            # Let's use the property: m_new = m_decayed + V_new * (r - x^T m_decayed) * x ?
            # Actually, standard ridge regression update:
            # m_new = m_old + V_new * (r - x^T m_old) * x
            # Let's verify.
            # m_new = A_new^{-1} (b_old + r x)
            # A_new m_new = b_old + r x
            # (A_old + x x^T) m_new = b_old + r x
            # A_old m_new + x (x^T m_new) = b_old + r x
            # m_old = A_old^{-1} b_old -> b_old = A_old m_old
            # A_old m_new + x (x^T m_new) = A_old m_old + r x
            # A_old (m_new - m_old) = x (r - x^T m_new)
            # (m_new - m_old) = A_old^{-1} x (r - x^T m_new)
            # This is implicit.
            # Easier: m_new = m_old + V_new * (r - x^T m_old) * x
            # Let's check:
            # m_new = m_old + V_new * x * (r - x^T m_old)
            # A_new m_new = (A_old + x x^T) (m_old + V_new x (r - x^T m_old))
            # = A_old m_old + A_old V_new x (r - x^T m_old) + x x^T m_old + x x^T V_new x (r - x^T m_old)
            # Note A_old V_new = A_old (A_old + x x^T)^{-1} = I - x (A_old + x x^T)^{-1} x^T = I - x V_new x^T
            # So A_old V_new x = x - x V_new x^T x = x (1 - V_new x^T x)
            # Term 2: x (1 - V_new x^T x) (r - x^T m_old)
            # Term 4: x (x^T V_new x) (r - x^T m_old)
            # Sum 2+4: x (r - x^T m_old)
            # Total: A_old m_old + x (r - x^T m_old) + x x^T m_old
            # = b_old + x r - x x^T m_old + x x^T m_old
            # = b_old + r x. Correct.
            
            m_new = m_decayed + V_new @ context * (reward - context @ m_decayed)
        
        # Ensure symmetry and positive definiteness for numerical stability
        V_new = (V_new + V_new.T) / 2.0
        # Add small regularization if needed
        V_new += np.eye(self.d) * 1e-6

        self.V[arm] = V_new
        self.m[arm] = m_new
# EVOLVE-BLOCK-END
