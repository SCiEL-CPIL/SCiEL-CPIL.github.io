    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation"""
        xm = x.copy()
        for i in range(len(xm)):
            if np.random.random() < self.p_m:
                delta1 = (xm[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - xm[i]) / (hi[i] - lo[i])
                rand = np.random.random()
                if rand < 0.5:
                    xq = 2.0 * rand + (1.0 - 2.0 * rand) * (1.0 + delta1) ** (self.eta_m + 1.0)
                    deltaq = (xq ** (1.0 / (self.eta_m + 1.0)) - 1.0)
                else:
                    xq = 2.0 * (1.0 - rand) + (2.0 * rand - 1.0) * (1.0 + delta2) ** (self.eta_m + 1.0)
                    deltaq = 1.0 - (xq ** (1.0 / (self.eta_m + 1.0)))
                xm[i] = xm[i] + deltaq * (hi[i] - lo[i])
        return np.clip(xm, lo, hi)
    