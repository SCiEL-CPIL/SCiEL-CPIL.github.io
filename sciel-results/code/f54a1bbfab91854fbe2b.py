# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Sparse Gaussian Process for non-stationary linear contextual bandits.

    Uses a sparse GP with inducing points to model the reward surface as a
    function of augmented features (context + one-hot arm encoding). This
    captures non-linear interactions between context and arm choice.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 118.0  # safety margin before 150s kill

        # Augmented dimension: context + one-hot arm
        self.d_aug = dim + n_arms

        # Sparse GP parameters
        self.n_inducing = 32  # number of inducing points
        self.inducing = np.random.randn(self.n_inducing, self.d_aug) * 0.5
        self.inducing_mean = np.zeros(self.d_aug)

        # GP hyperparameters
        self.length_scale = 1.0
        self.signal_var = 1.0
        self.noise_var = 0.1
        self.alpha_gp = 1.0  # exploration constant for UCB

        # Data buffer for online updates
        self.X_data = []  # list of augmented feature vectors
        self.y_data = []  # list of rewards
        self.max_data = 200  # max data points to keep

        # For fast computation
        self._use_fast = False

    def _augment(self, context, arm):
        """Create augmented feature vector: [context, one-hot(arm)]"""
        context = np.asarray(context, dtype=np.float64)
        x = np.zeros(self.d_aug)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def _rbf_kernel(self, X1, X2):
        """RBF kernel between two sets of points."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dists = (
            np.sum(X1**2, axis=1)[:, None] +
            np.sum(X2**2, axis=1)[None, :] -
            2 * X1 @ X2.T
        )
        sq_dists = np.maximum(sq_dists, 0)
        return self.signal_var * np.exp(-sq_dists / (2 * self.length_scale**2))

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            # Fallback: use simple linear model if GP is too slow
            scores = []
            for a in range(self.n):
                x = self._augment(context, a)
                if len(self.X_data) > 0:
                    # Simple nearest-neighbor prediction
                    X = np.array(self.X_data)
                    y = np.array(self.y_data)
                    dists = np.sum((X - x[None, :])**2, axis=1)
                    k = min(5, len(X))
                    idx = np.argsort(dists)[:k]
                    pred = np.mean(y[idx])
                else:
                    pred = 0.0
                scores.append(pred)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            x = self._augment(context, a)
            if len(self.X_data) == 0:
                scores.append(0.0)
                continue

            X = np.array(self.X_data)
            y = np.array(self.y_data)
            F = self._rbf_kernel(x[None, :], self.inducing[None, :])[0]
            K_ii = self._rbf_kernel(self.inducing, self.inducing) + self.noise_var * np.eye(self.n_inducing)
            K_yy = self._rbf_kernel(X, X) + self.noise_var * np.eye(len(X))
            
            # Sparse GP prediction
            # m = F @ (K_ii^{-1} @ (F^T @ (K_yy^{-1} @ y)))
            # v = F @ (K_ii^{-1} @ (F^T - (K_ii^{-1} @ F^T @ K_yy) @ K_yy^{-1} @ F)) @ K_ii^{-1} @ F^T
            
            try:
                K_ii_inv = np.linalg.inv(K_ii)
                F_X = self._rbf_kernel(self.inducing, X)
                K_yy_inv = np.linalg.inv(K_yy)
                
                # Mean prediction
                m = F @ (K_ii_inv @ (F_X.T @ (K_yy_inv @ y)))
                
                # Variance prediction (simplified)
                v = self.signal_var - F @ K_ii_inv @ F.T
                v = max(v, 0.0)
                
                ucb = m + self.alpha_gp * np.sqrt(v)
                scores.append(ucb)
            except np.linalg.LinAlgError:
                # Fallback to simple mean
                scores.append(np.mean(y))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        x = self._augment(context, arm)

        # Add to data buffer
        self.X_data.append(x)
        self.y_data.append(reward)
        if len(self.X_data) > self.max_data:
            self.X_data.pop(0)
            self.y_data.pop(0)

        self.t += 1

        # Periodically adapt hyperparameters based on residual
        if self.t % 10 == 0 and len(self.y_data) >= 10:
            residuals = []
            for i, (xi, yi) in enumerate(zip(self.X_data, self.y_data)):
                if i == len(self.X_data) - 1:
                    continue  # skip current point
                if len(self.X_data) < 2:
                    break
                # Simple prediction using nearest neighbors
                X = np.array(self.X_data[:i])
                y = np.array(self.y_data[:i])
                dists = np.sum((X - xi[None, :])**2, axis=1)
                k = min(3, len(X))
                idx = np.argsort(dists)[:k]
                pred = np.mean(y[idx])
                residuals.append(abs(yi - pred))
            
            if residuals:
                mean_res = np.mean(residuals)
                # Adapt noise variance
                self.noise_var = max(0.01, min(1.0, mean_res**2 * 0.5))
                # Adapt length scale based on data spread
                if len(self.X_data) > 5:
                    X = np.array(self.X_data)
                    spread = np.std(X, axis=0)
                    self.length_scale = max(0.1, min(10.0, np.mean(spread)))

        # Adapt inducing points to better cover data space
        if self.t % 50 == 0 and len(self.X_data) >= 10:
            X = np.array(self.X_data)
            # Keep some random inducing points and add some data points
            n_keep = self.n_inducing // 2
            self.inducing = np.vstack([
                self.inducing[:n_keep],
                X[np.random.choice(len(X), size=self.n_inducing - n_keep, replace=False)]
            ])
# EVOLVE-BLOCK-END
