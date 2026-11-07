            # Update covariance (Rank-Mu only, safer)
            ydiff = np.array([xs[i] - x_mean for i in range(mu)])
            C_new = (1 - c_sigma) * C
            for i in range(mu):
                C_new += c_sigma * w[i] * np.outer(ydiff[i], ydiff[i])
            C_new = (C_new + C_new.T) / 2
