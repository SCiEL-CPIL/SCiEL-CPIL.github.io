import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Strategy:
    Uses a single global Bayesian Linear Regression model with Thompson Sampling.
    Unlike per-arm models, this leverages the fact that in a linear contextual bandit,
    the reward function is shared across contexts (and implicitly arms if arms are part of context or if the context captures arm-specific effects). 
    However, the interface `select(context)` returns an arm index, implying the context might be arm-agnostic or the arm is chosen based on context.
    Wait, the interface is `select(context) -> arm`. This implies the context is given, and we pick an arm. 
    In standard linear bandits, the reward is $x^T w_a + \epsilon$ where $x$ is context and $w_a$ is arm-specific weight.
    Or, if it's a "linear contextual bandit" where the context includes the arm? No, `select` returns the arm.
    So the model must be per-arm: $r = x^T w_a + \epsilon$.
    
    The current program uses per-arm models. My "different strategy" should be materially different.
    Let's try a **Global Contextual Model with Arm Embedding** or **Thompson Sampling with Adaptive Forgetting**.
    
    Actually, a very different approach for non-stationary linear bandits is **Sliding Window** or **Discounted** variants, but the current program already does adaptive forgetting.
    
    Let's try **Thompson Sampling (TS)** instead of UCB. TS is often superior in non-stationary settings because it naturally balances exploration/exploitation based on the posterior uncertainty, which can be updated more dynamically.
    
    Key Difference:
    1. Use Thompson Sampling: Sample weights from the posterior distribution $N(\theta_t, \sigma^2 A_t^{-1})$ and pick the arm with the highest sampled reward.
    2. Adaptive Forgetting: Instead of per-arm error-driven forgetting, use a **Global Change Detection** mechanism that monitors the variance of recent rewards or prediction errors to trigger a "reset" or aggressive decay of the global model (or per-arm models if we keep them).
    
    Let's stick to per-arm models because the reward is arm-specific ($w_a$).
    But let's switch from UCB to **Thompson Sampling**.
    
    Additionally, instead of a complex adaptive alpha per arm, let's use a **Fixed but Aggressive Forgetting Factor** combined with **Thompson Sampling**. 
    Or better: **Adaptive Forgetting based on a Global Non-stationarity Signal** that applies a uniform decay to all arms, but using a simpler, more robust signal like the difference between the best and worst arm rewards or the variance of recent rewards.
    
    Let's implement:
    1. Per-arm Bayesian Linear Regression (A, b).
    2. Thompson Sampling for selection: Sample $\theta_a \sim N(\hat{\theta}_a, \sigma^2 A_a^{-1})$, pick $a$ maximizing $x^T \theta_a$.
    3. Adaptive Forgetting: 
       - Maintain a global "surprise" signal based on the reward of the chosen arm vs its predicted mean.
       - If surprise is high, decrease alpha (forget faster).
       - To be materially different from the current program's "local + global min" logic, let's use a **Change Point Detection** heuristic: 
         If the moving average of rewards drops or the variance spikes, trigger a **Hard Reset** (or near-reset) of the models. This is a discrete strategy vs the continuous adaptive decay of the current program.
    
    Actually, a hard reset is risky. Let's use **Thompson Sampling** with a **Fixed Forgetting Factor** that is slightly more aggressive than the current base, but the key is the TS policy. The current program uses UCB with a dynamic alpha. TS with a fixed alpha might be more robust and less prone to over-exploration/under-exploration oscillations.
    
    Let's refine: 
    - Use Thompson Sampling.
    - Use a single, global adaptive forgetting factor $\alpha_t$.
    - $\alpha_t$ is updated based on the **prediction error** of the chosen arm.
    - If error is large, $\alpha$ drops (faster forgetting).
    - If error is small, $\alpha$ rises (slower forgetting).
    - This applies the same $\alpha$ to all arms in each step (or just the chosen one? Usually, you decay all models by $\alpha$ at each step, or just the chosen one? In standard discounted linear bandits, you decay all models by $\alpha$ at each step to account for the passage of time and potential drift in all arms).
    
    Let's implement **Discounted Thompson Sampling** with **Global Adaptive Forgetting**.
    
    Steps:
    1. Init: A_a = I, b_a = 0.
    2. Select:
       - For each arm a:
         - $\theta_a = A_a^{-1} b_a$
         - $\Sigma_a = A_a^{-1}$
         - Sample $\hat{\theta}_a \sim N(\theta_a, \sigma^2 \Sigma_a)$
         - $s_a = x^T \hat{\theta}_a$
       - Pick arm with max $s_a$.
    3. Update:
       - Compute prediction error $e = r - x^T \theta_{a^*}$.
       - Update global error EMA.
       - Compute new $\alpha$ based on error EMA.
       - For ALL arms:
         - $A_a = \alpha A_a + xx^T$ (if a == a*) else $A_a = \alpha A_a$
         - $b_a = \alpha b_a + r x$ (if a == a*) else $b_a = \alpha b_a$
         - Note: Only the chosen arm gets new data. All arms get decayed. This is the standard discounted approach.
    
    This is different from the current program which:
    - Uses UCB (deterministic score with bonus).
    - Uses per-arm adaptive alpha (local signal).
    - Only decays the chosen arm with adaptive alpha, and conditionally decays others.
    
    My approach:
    - Uses TS (stochastic sample).
    - Uses global adaptive alpha (single signal).
    - Decays ALL arms every step.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Thompson Sampling noise variance
        self.sigma = 1.0
        
        # Adaptive forgetting parameters
        self.base_alpha = 0.99
        self.min_alpha = 0.90
        self.max_alpha = 0.999
        
        # Global error tracking
        self.error_ema = 0.0
        self.error_ema_alpha = 0.2
        self.error_scale = 1.0 # Used to normalize error to [0,1] range roughly
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def select(self, context):
        if time.time() > self.deadline:
            return 0
        
        context = np.asarray(context)
        scores = []
        
        for a in range(self.n):
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
                
                # Sample from posterior
                # Covariance is sigma^2 * inv_A
                # We can use np.random.multivariate_normal or manual sampling
                # For dim=5, manual is fast:
                # L = cholesky(inv_A)
                # z = normal(0, I)
                # theta_sample = theta + sigma * L @ z
                
                L = np.linalg.cholesky(inv_A)
                z = np.random.normal(0, 1, self.d)
                theta_sample = theta + self.sigma * (L @ z)
                
                score = theta_sample @ context
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            inv_A = np.linalg.inv(self.A[arm])
            theta = inv_A @ self.b[arm]
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_abs = abs(error)
        except Exception:
            error_abs = 1.0

        # 2. Update Global Error EMA
        self.error_ema = (1 - self.error_ema_alpha) * self.error_ema + self.error_ema_alpha * error_abs
        
        # 3. Determine adaptive forgetting factor
        # Map error to alpha.
        # We want high error -> low alpha (fast forgetting).
        # Normalize error. If error is 0, alpha = max. If error > 1.0, alpha = min.
        # Simple linear mapping with clipping.
        err_norm = min(max(self.error_ema / self.error_scale, 0.0), 1.0)
        alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm
        
        # 4. Update Models
        # Decay all arms
        for a in range(self.n):
            self.A[a] = alpha * self.A[a]
            self.b[a] = alpha * self.b[a]
            
            if a == arm:
                self.A[a] += np.outer(context, context)
                self.b[a] += reward * context

        # 5. Time guard
        if time.time() > self.deadline:
            pass
