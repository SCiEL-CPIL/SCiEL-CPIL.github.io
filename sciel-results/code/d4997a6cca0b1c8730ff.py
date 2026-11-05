<<<<<<< SEARCH
    def select(self, context):
        # Deadline guard
        if time.time() - self.t0 > 110:
            return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))

        scores = []
        for a in range(self.n):
            # Predict step
            P = self._kalman_predict(a)
            theta = self.theta[a]

            # Mean prediction
            mean = theta @ context

            # Uncertainty
            var = context @ P @ context + self.R
            std = np.sqrt(max(var, 1e-8))

            # UCB-style score with adaptive exploration
            # More exploration in volatile regime
            alpha = 0.5 if self.regime[a] == 0 else 1.0
            score = mean + alpha * std
            scores.append(score)

        return int(np.argmax(scores))
=======
    def select(self, context):
        # Deadline guard
        if time.time() - self.t0 > 110:
            return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))

        scores = []
        for a in range(self.n):
            # Predict step
            P = self._kalman_predict(a)
            theta = self.theta[a]

            # Check for NaN in theta
            if not np.all(np.isfinite(theta)):
                scores.append(-np.inf)
                continue

            # Mean prediction
            mean = theta @ context

            # Uncertainty
            var = context @ P @ context + self.R
            std = np.sqrt(max(var, 1e-8))

            # Adaptive exploration: more exploration in volatile regime
            # Also scale by how uncertain we are overall
            base_alpha = 0.6 if self.regime[a] == 0 else 1.2
            # Slightly increase exploration early on
            if self.round < 50:
                base_alpha *= 1.2
            score = mean + base_alpha * std
            scores.append(score)

        return int(np.argmax(scores))
>>>>>>> REPLACE

