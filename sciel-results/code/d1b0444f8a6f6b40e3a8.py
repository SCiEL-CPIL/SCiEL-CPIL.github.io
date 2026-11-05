<<<<<<< SEARCH
    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error before updating
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        self.prediction_errors[arm].append(err)
        if len(self.prediction_errors[arm]) > self.max_err_window:
            self.prediction_errors[arm].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
=======
    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Augment context with one-hot arm indicator
        x_aug = np.zeros(self.total_dim)
        x_aug[:self.d] = context
        x_aug[self.d + arm] = 1.0

        # Compute prediction error before updating
        theta = self.V @ self.b
        pred = theta @ x_aug
        err = abs(reward - pred)
        self.prediction_errors.append(err)
        if len(self.prediction_errors) > self.max_err_window:
            self.prediction_errors.pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V
        Vx = V_old @ x_aug
        xVx = x_aug @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V = np.eye(self.total_dim) * self.lambda_reg

        # Update b with forgetting
        self.b = gamma * self.b + reward * x_aug
>>>>>>> REPLACE
