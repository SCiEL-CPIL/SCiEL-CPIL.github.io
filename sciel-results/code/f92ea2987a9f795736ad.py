        for i in range(n):
            dx = pts[i, 0] - pts[i + 1:, 0]
            dy = pts[i, 1] - pts[i + 1:, 1]
            d = np.hypot(dx, dy)
            # Avoid division by zero
            d = np.where(d < 1e-10, 1e-10, d)
            # Force magnitude: 1/d^2 (strong when close)
            f_mag = repel_strength / (d * d)
            fx = f_mag * dx / d
            fy = f_mag * dy / d
            forces[i, 0] += fx.sum()
            forces[i, 1] += fy.sum()
            # Apply equal and opposite to others
            forces[i + 1:, 0] -= fx
            forces[i + 1:, 1] -= fy
