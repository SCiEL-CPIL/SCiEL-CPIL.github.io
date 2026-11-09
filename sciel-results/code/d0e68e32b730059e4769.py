"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Kernelized Linear Bandit (KLB)
    with RBF kernel and Exponential Forgetting in the Dual Formulation.
    
    Strategy:
    1. Kernelization: Maps contexts to a high-dimensional feature space using an RBF kernel.
       This allows the model to capture non-linear relationships between context and reward.
    2. Dual Formulation: Instead of maintaining weight vectors in the feature space (infinite dimensional),
       we maintain weights on the observed data points (dual variables).
       Prediction for x is: y_hat = sum_i alpha_i * K(x, x_i)
    3. Exponential Forgetting: The dual variables are updated with a discount factor gamma < 1.
       This effectively gives more weight to recent observations, allowing the model to adapt
       to non-stationary changes in arm weights.
    4. Thompson Sampling: We sample from the posterior distribution of the weights to balance
       exploration and exploitation.
    
    Mathematical Formulation:
    Let D_t = {(x_1, y_1), ..., (x_t, y_t)} be the history.
    The posterior mean of the weight vector w in the RKHS is:
    w_t = sum_{i=1}^t alpha_i x_i
    where alpha = K_t^{-1} y_t (with regularization).
    
    With exponential forgetting, we approximate the inverse kernel matrix update or simply
    decay the influence of old samples. A common efficient approximation for non-stationary KL
    is to maintain a decaying sum of outer products of kernel vectors.
    
    However, a more direct and robust approach for non-stationary linear bandits in the dual form
    is to use a discounted kernel ridge regression.
    
    Let's use the standard Kernel Ridge Regression with a discount factor.
    Objective: min_w ||w||_K^2 + lambda * sum_t gamma^t ||y_t - <w, phi(x_t)>||^2
    
    The solution is w = sum_i alpha_i phi(x_i)
    alpha = (K + lambda * D)^{-1} y
    where D is a diagonal matrix with entries gamma^t.
    
    Since inverting K+lambda*D is expensive, we can use an online update or a sliding window of dual variables.
    Given the small number of arms (5) and the need for speed, we will maintain a decaying history of dual coefficients.
    
    Actually, a simpler and very effective method for non-stationary linear bandits is **Discounted Linear Bandit** 
    in the primal, but the prompt asks for a DIFFERENT strategy. The current program uses sliding window linear.
    
    Let's implement **Kernelized Linear Bandit** with **Exponential Forgetting** on the sufficient statistics.
    We maintain P_t and theta_t for each arm, but in the feature space? No, that's infinite.
    
    We will use the **Dual** variables with exponential decay.
    For each arm a, we maintain a set of recent samples (x_i, y_i) and their associated dual weights alpha_i.
    To handle non-stationarity, we decay the alpha_i by a factor gamma each step.
    Prediction: y_hat = sum alpha_i K(x, x_i)
    Variance for TS: We need the posterior covariance.
    
    This can be computationally heavy if we store many points. 
    Alternative: Use a **Fixed-Dimension Kernel Approximation** (Random Fourier Features for RBF) to project contexts into a finite, high-dimensional space, then apply standard Discounted Linear Bandit (RLS) in that space.
    This is effectively "Kernelized" via approximation and allows O(D^2) updates where D is the approximation dimension.
    
    Let's use Random Fourier Features (RFF) to map x -> z(x) in R^D.
    Then use Discounted Linear Regression (RLS) with discount factor gamma on z(x).
    This is a "Kernelized" approach in spirit (approximating the kernel) and uses a different mechanism (RFF + RLS) than the current (Sliding Window + Least Squares).
    
    Steps:
    1. Generate RFF mapping parameters.
    2. Map context x to z(x).
    3. Maintain P (inverse covariance) and theta for each arm.
    4. Update P and theta using RLS with discount factor gamma.
    5. Use Thompson Sampling: sample theta ~ N(theta_hat, P).
    6. Score = theta_sample @ z(x).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # RFF Parameters
        # D is the dimension of the feature space. 
        # Larger D approximates RBF better but increases compute.
        # dim is usually small (e.g., 5-10). D=64 or 128 is reasonable.
        self.D = 128 
        # Bandwidth sigma for RBF. 
        # Heuristic: 1 / sqrt(dim) or similar. Let's use 1.0 for now, can be tuned.
        self.sigma = 1.0
        
        # Generate RFF weights and biases
        # w ~ N(0, 1/sigma^2), b ~ U(0, 2*pi)
        self.rff_weights = np.random.randn(self.D, self.d) / self.sigma
        self.rff_biases = np.random.uniform(0, 2 * np.pi, self.D)
        
        # Discount factor for non-stationarity
        # gamma close to 1 means slow forgetting. 
        # 0.95 or 0.98 is typical. 
        # Let's use 0.95 for faster adaptation.
        self.gamma = 0.95
        
        # Regularization
        self.lam = 1.0
        
        # Initialize P and theta for each arm
        # P is D x D, theta is D x 1
        self.P = [np.eye(self.D) / self.lam for _ in range(n_arms)]
        self.theta = [np.zeros((self.D, 1)) for _ in range(n_arms)]
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _map_context(self, context):
        """
        Map context to RFF feature space.
        z(x) = sqrt(2/D) * cos(w @ x + b)
        """
        context = np.asarray(context, dtype=float)
        # (D, d) @ (d,) -> (D,)
        z = np.cos(self.rff_weights @ context + self.rff_biases) * np.sqrt(2.0 / self.D)
        return z

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: Greedy using current means
            best_arm = 0
            best_score = -np.inf
            z = self._map_context(context)
            for a in range(self.n):
                score = self.theta[a].T @ z
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        z = self._map_context(context)
        scores = []
        
        for a in range(self.n):
            # Thompson Sampling
            # Sample theta_sample ~ N(theta_a, P_a)
            # We can do this by: theta_sample = theta_a + L @ z_noise, where L is Cholesky of P_a
            # P_a is symmetric positive definite.
            
            try:
                L = np.linalg.cholesky(self.P[a])
            except np.linalg.LinAlgError:
                # Add jitter if not positive definite
                self.P[a] += 1e-6 * np.eye(self.D)
                try:
                    L = np.linalg.cholesky(self.P[a])
                except np.linalg.LinAlgError:
                    # Fallback to mean only
                    L = np.zeros((self.D, self.D))
            
            z_noise = np.random.randn(self.D)
            theta_sample = self.theta[a].flatten() + L @ z_noise
            
            score = theta_sample @ z
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        z = self._map_context(context)
        reward = np.array([[reward]])
        
        # RLS Update with Discounting
        # P_t = (1/gamma) * P_{t-1}
        # Then standard RLS update:
        # K_t = P_t z / (lambda + z^T P_t z)
        # P_{t+1} = (P_t - K_t z^T P_t) / lambda  (Note: lambda handling varies by formulation)
        
        # Let's use the standard Discounted RLS formulation:
        # Minimize: sum_{i=1}^t gamma^{t-i} (y_i - theta^T z_i)^2 + lambda ||theta||^2
        # This is equivalent to updating P and theta as follows:
        # P_t = (1/gamma) P_{t-1}
        # K_t = P_t z_t / (1 + z_t^T P_t z_t)  (if lambda is absorbed or handled separately)
        
        # A more robust formulation for Discounted RLS:
        # P_t = (1/gamma) P_{t-1}
        # Denom = 1 + z_t^T P_t z_t
        # K_t = P_t z_t / Denom
        # theta_t = theta_{t-1} + K_t (reward - z_t^T theta_{t-1})
        # P_{t+1} = (P_t - K_t z_t^T P_t) / lambda  (if we have explicit regularization)
        
        # Actually, the standard RLS with discounting is:
        # P_t = (1/gamma) P_{t-1}
        # K_t = P_t z_t / (1 + z_t^T P_t z_t)
        # theta_t = theta_{t-1} + K_t (y_t - z_t^T theta_{t-1})
        # P_{t+1} = (P_t - K_t z_t^T P_t)
        
        # We need to handle the initial regularization lambda.
        # P_0 = I / lambda.
        # The update above assumes the prior is handled by P.
        
        P_a = self.P[arm]
        theta_a = self.theta[arm].flatten()
        
        # 1. Discount P
        P_a = P_a / self.gamma
        
        # 2. Compute Kalman Gain
        Pz = P_a @ z
        denom = 1.0 + z @ Pz
        K = Pz / denom
        
        # 3. Update theta
        prediction = z @ theta_a
        error = reward - prediction
        theta_a = theta_a + K * error
        
        # 4. Update P
        # P_new = P_a - K z^T P_a
        # To maintain symmetry and positive definiteness, we can use:
        # P_new = (I - K z^T) P_a
        # But for numerical stability, we might just do the direct update.
        # P_a is D x D. K is D x 1. z is D x 1.
        # P_new = P_a - np.outer(K, z) @ P_a
        
        P_a = P_a - np.outer(K, z) @ P_a
        
        # Ensure symmetry
        P_a = (P_a + P_a.T) / 2.0
        
        # Store back
        self.P[arm] = P_a
        self.theta[arm] = theta_a.reshape(-1, 1)
        
        self.t += 1
# EVOLVE-BLOCK-END
