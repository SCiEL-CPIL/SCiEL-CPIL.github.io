# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Adaptive Discounted Least Squares.
    
    Key improvements:
    1. Residual-Driven Forgetting: beta adapts to recent prediction error.
    2. Refined Exploration: alpha modulated by non-stationarity and step count.
    3. Robust Linear Algebra: Cholesky with fallback.
    4. Time Guard: Ensures completion within wall-clock limits.
    5. Improved numerical stability and tuning.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization

        # Per-arm model parameters
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Statistics for adaptive beta
        self.residual_ema = 0.0
        self.residual_alpha = 0.1  # EMA smoothing factor
        
        # Beta bounds - slightly tighter range for stability
        self.beta_min = 0.82
        self.beta_max = 0.99
        
        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_nonstat_weight = 1.2
        self.alpha_early_boost = 0.8  # extra exploration when few observations
        
        # Step counter for early-phase exploration
        self.step = 0
        self.early_phase_steps = 30

        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _get_adaptive_beta(self):
        """
        Determine the discount factor beta based on recent prediction errors.
        High residual error -> lower beta (faster forgetting).
        Low residual error -> higher beta (slower forgetting).
        """
        r = self.residual_ema
        low_r = 0.05
        high_r = 1.5
        if r <= low_r:
            nonstat_score = 0.0
        elif r >= high_r:
            nonstat_score = 1.0
        else:
            nonstat_score = (r - low_r) / (high_r - low_r)
        
        # Interpolate beta between beta_max and beta_min
        beta = self.beta_max - nonstat_score * (self.beta_max - self.beta_min)
        return beta, nonstat_score

    def _solve_theta(self, a):
        """Solve for theta using Cholesky decomposition for speed and stability."""
        try:
            L = np.linalg.cholesky(self.A[a])
            y = np.linalg.solve(L, self.b[a])
            theta = np.linalg.solve(L.T, y)
            return theta, L
        except np.linalg.LinAlgError:
            try:
                theta = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]
                return theta, None
            except Exception:
                return np.zeros(self.d), None

    def select(self, context):
        if self._safe():
            # In emergency, pick the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta, _ = self._solve_theta(a)
                    scores.append(float(theta @ context))
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))

        beta, nonstat_score = self._get_adaptive_beta()
        
        # Adaptive alpha: increase exploration if non-stationarity is detected
        # Also increase exploration early on (first few steps)
        early_factor = 1.0 + self.alpha_early_boost * max(0.0, (self.early_phase_steps - self.step) / self.early_phase_steps)
        
        # Base alpha * early boost + non-stationarity boost
        alpha = self.alpha_base * early_factor + self.alpha_nonstat_weight * nonstat_score
        
        scores = []
        for a in range(self.n):
            try:
                theta, L = self._solve_theta(a)
                pred = float(theta @ context)
                
                if L is not None:
                    # Variance term: context^T A^-1 context
                    v = np.linalg.solve(L, context)
                    var_term = float(v @ v)
                else:
                    # Fallback: approximate variance
                    try:
                        cov = np.linalg.inv(self.A[a])
                        var_term = float(context @ cov @ context)
                    except Exception:
                        var_term = 0.0
                
                ucb_score = pred + alpha * np.sqrt(max(var_term, 0.0))
                scores.append(ucb_score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.step += 1
        
        # Compute prediction error to update residual EMA
        try:
            theta, _ = self._solve_theta(arm)
            pred = float(theta @ context)
            residual = reward - pred
            residual_sq = residual * residual
            
            # Update EMA of squared residuals
            self.residual_ema = (1 - self.residual_alpha) * self.residual_ema + self.residual_alpha * residual_sq
        except Exception:
            pass

        beta, _ = self._get_adaptive_beta()

        # Update the model for the chosen arm
        # Discounted Least Squares update with numerical stability guard
        new_A = beta * self.A[arm] + np.outer(context, context)
        # Ensure positive definiteness
        try:
            eig_min = np.min(np.linalg.eigvalsh(new_A))
            if eig_min < 1e-6:
                new_A += (1e-6 - eig_min) * np.eye(self.d)
        except Exception:
            pass
        
        self.A[arm] = new_A
        self.b[arm] = beta * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
