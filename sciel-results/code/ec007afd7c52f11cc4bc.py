"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Initial layout: hexagonal packing with 4 columns of 7,6,7,6 circles
    # Column x-positions: 0.5*dx, 1.5*dx, 2.5*dx, 3.5*dx
    # Row y-positions: (j+0.5)*dy for j in range(n_j)
    # Hexagonal offset: odd columns shifted by dy/2

    dx = np.sqrt(3) / 14.0  # column spacing
    dy = 1.0 / 7.0           # row spacing
    r_init = dx / 2.0        # radius (half column spacing)

    # Build initial positions
    cols = [7, 6, 7, 6]
    positions = []
    radii = []
    for c, n_j in enumerate(cols):
        x = (c + 0.5) * dx
        for j in range(n_j):
            y = (j + 0.5) * dy
            if c % 2 == 1:  # offset odd columns
                y += dy / 2.0
                y = min(y, 1.0 - r_init)  # clamp
                y = max(y, r_init)
            positions.append([x, y])
            radii.append(r_init)

    positions = np.array(positions, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)

    def evaluate(pos, rad):
        """Compute sum of radii and check validity."""
        # Check boundaries
        if np.any(pos[:, 0] - rad < -1e-10) or np.any(pos[:, 0] + rad > 1.0 + 1e-10):
            return None
        if np.any(pos[:, 1] - rad < -1e-10) or np.any(pos[:, 1] + rad > 1.0 + 1e-10):
            return None
        # Check overlaps
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                if dist < rad[i] + rad[j] - 1e-10:
                    return None
        return np.sum(rad)

    best_score = evaluate(positions, radii)
    if best_score is None:
        best_score = 0.0
    best_pos = positions.copy()
    best_rad = radii.copy()

    # Gradient descent with penalty method
    # We'll optimize positions and radii jointly
    # Penalty: for each pair, penalize if dist < r_i + r_j
    # Penalty: for each circle, penalize if outside [0,1]^2

    def compute_gradients(pos, rad, penalty_weight):
        """Compute gradients of (sum radii - penalty) w.r.t. pos and rad."""
        n = len(pos)
        dpos = np.zeros_like(pos)
        drad = np.ones_like(rad)  # gradient of sum(rad) w.r.t. rad

        # Pairwise overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff**2)) + 1e-12
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    # Penalty term: penalty_weight * overlap^2
                    # d/dpos_i = -penalty_weight * 2 * overlap * diff / dist
                    # d/drad_i = -penalty_weight * 2 * overlap
                    grad_pos = penalty_weight * 2.0 * overlap * diff / dist
                    dpos[i] += grad_pos
                    dpos[j] -= grad_pos
                    drad[i] -= penalty_weight * 2.0 * overlap
                    drad[j] -= penalty_weight * 2.0 * overlap

        # Boundary penalties
        for i in range(n):
            # Left: x - r < 0
            viol = rad[i] - pos[i, 0]
            if viol > 0:
                dpos[i, 0] += penalty_weight * 2.0 * viol
                drad[i] -= penalty_weight * 2.0 * viol
            # Right: x + r > 1
            viol = pos[i, 0] + rad[i] - 1.0
            if viol > 0:
                dpos[i, 0] -= penalty_weight * 2.0 * viol
                drad[i] -= penalty_weight * 2.0 * viol
            # Bottom: y - r < 0
            viol = rad[i] - pos[i, 1]
            if viol > 0:
                dpos[i, 1] += penalty_weight * 2.0 * viol
                drad[i] -= penalty_weight * 2.0 * viol
            # Top: y + r > 1
            viol = pos[i, 1] + rad[i] - 1.0
            if viol > 0:
                dpos[i, 1] -= penalty_weight * 2.0 * viol
                drad[i] -= penalty_weight * 2.0 * viol

        return dpos, drad

    # Adaptive penalty gradient descent
    lr = 0.01
    penalty_weight = 100.0
    max_iters = 5000

    for it in range(max_iters):
        if time.time() > deadline:
            break

        dpos, drad = compute_gradients(positions, radii, penalty_weight)

        # Update
        positions += lr * dpos
        radii += lr * drad

        # Clamp radii to positive
        radii = np.maximum(radii, 0.01)

        # Clamp positions to reasonable range
        positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
        positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)

        # Evaluate
        score = evaluate(positions, radii)
        if score is not None and score > best_score:
            best_score = score
            best_pos = positions.copy()
            best_rad = radii.copy()

        # Adaptive learning rate: reduce if oscillating
        if it % 100 == 0 and it > 0:
            # Check if we're making progress
            pass

        # Reduce learning rate gradually
        if it % 500 == 499:
            lr *= 0.8
            penalty_weight *= 1.2

    # Final polish: try to increase radii uniformly
    # Scale all radii by a factor and check validity
    scale = 1.0
    for _ in range(200):
        if time.time() > deadline:
            break
        new_rad = best_rad * (1.0 + 0.001)
        if evaluate(best_pos, new_rad) is not None:
            best_rad = new_rad
        else:
            break

    # Return the best valid configuration
    result = np.column_stack([best_pos, best_rad])
    return result


if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(result)
