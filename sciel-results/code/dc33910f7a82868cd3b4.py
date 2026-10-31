"""Circle packing in perimeter-4 rectangle: Kinetic soft-body relaxation."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def run_simulation(z0, time_budget):
        """Run kinetic relaxation simulation. Returns (x, y, r, w) or None."""
        z0 = np.array(z0, dtype=float)
        n = N
        x = z0[0::3][:n].copy()
        y = z0[1::3][:n].copy()
        r = z0[2::3][:n].copy()
        w = z0[-1]

        vx = np.zeros(n)
        vy = np.zeros(n)
        vr = np.zeros(n)
        vw = 0.0

        # Parameters
        dt = 0.002
        damping = 0.96
        k_rep = 300.0
        k_wall = 400.0
        k_attract = 0.00005
        k_aspect = 0.05
        r_min = 0.01
        r_max = 0.5
        w_min = 0.3
        w_max = 1.7

        max_iter = int(time_budget * 1000)

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            h = 2.0 - w
            if h < 0.1 or w < 0.1:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)
            fr = np.zeros(n)
            fw = 0.0

            # 1. Pairwise repulsion (Coulomb-like, soft)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            d = np.sqrt(np.maximum(d_sq, 1e-12))
            r_sum = r[:, None] + r[None, :]

            # Force magnitude: k / d^2, active when d < r_sum + margin
            margin = 0.1
            active = d < (r_sum + margin)
            # Avoid division by zero
            d_safe = np.where(d > 1e-6, d, 1e-6)
            f_mag = k_rep / (d_safe * d_safe)
            f_mag = np.where(active, f_mag, 0.0)

            # Direction
            ux = dx / d_safe
            uy = dy / d_safe

            # Sum forces (symmetric)
            fx = np.sum(f_mag * ux, axis=1)
            fy = np.sum(f_mag * uy, axis=1)

            # 2. Wall repulsion
            # Left
            viol = np.maximum(r - x, 0.0)
            fx -= viol * k_wall
            # Right
            viol = np.maximum(x + r - w, 0.0)
            fx += viol * k_wall
            # Bottom
            viol = np.maximum(r - y, 0.0)
            fy -= viol * k_wall
            # Top
            viol = np.maximum(y + r - h, 0.0)
            fy += viol * k_wall

            # 3. Attraction to grow radii (weak constant force)
            fr[:] = k_attract

            # 4. Aspect ratio penalty (keep w near 1.0)
            fw += k_aspect * (1.0 - w)

            # 5. Integrate velocities
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            vr = vr * damping + fr * dt
            vw = vw * damping + fw * dt

            # Update positions
            x += vx * dt
            y += vy * dt
            r += vr * dt
            w += vw * dt

            # Clamp constraints (project back)
            x = np.clip(x, 0.0, 2.0)
            y = np.clip(y, 0.0, 2.0)
            r = np.clip(r, r_min, r_max)
            w = np.clip(w, w_min, w_max)

            # Ensure circles don't go outside due to large steps
            h = 2.0 - w
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

            # If all velocities are small, we might be converged
            # But we continue to allow slow drift
            if it % 500 == 0 and it > 1000:
                # Check for stability
                if np.max(np.abs(vx)) < 1e-5 and np.max(np.abs(vy)) < 1e-5:
                    # Try to break symmetry by adding small noise
                    rng = np.random.default_rng(it)
                    x += rng.uniform(-0.001, 0.001, n)
                    y += rng.uniform(-0.001, 0.001, n)
                    r += rng.uniform(-0.0005, 0.0005, n)
                    x = np.clip(x, r, w - r)
                    y = np.clip(y, r, h - r)

        # Final polish: ensure strict validity
        # If invalid, shrink radii slightly
        if not _is_valid(x, y, r, w, h):
            for scale in [0.99, 0.98, 0.95, 0.90]:
                r_trial = r * scale
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break
            else:
                return None

        if not _is_valid(x, y, r, w, h):
            return None

        val = float(np.sum(r))
        return x, y, r, w, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Structured starts
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.4
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.1)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Allocate time budget per guess
    # We have ~145s. 20 guesses. ~7s each.
    # But we want to spend more time on better guesses if possible.
    # Strategy: Run all, keep best.
    time_per_guess = (deadline - start_time) / len(guesses)
    # Cap time per guess to avoid one taking too long, but allow more if few guesses
    time_per_guess = min(time_per_guess, 10.0)

    for i, z0 in enumerate(guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break

        # Dynamic budget: use remaining time divided by remaining guesses
        remaining_guesses = len(guesses) - i
        budget = (deadline - 2.0 - elapsed) / remaining_guesses
        budget = max(2.0, min(budget, 12.0))

        try:
            res = run_simulation(z0, budget)
            if res is not None:
                x, y, r, w, val = res
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            continue

    # Final polish on best result: try to expand radii further
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w = best_result
        h = 2.0 - w

        # Local search: try to nudge radii up
        # Simple gradient ascent on sum(r) with validity check
        # Since we can't easily compute gradients of validity, we use random perturbation
        # Or just try uniform scaling
        for scale in np.linspace(1.001, 1.05, 50):
            r_trial = r * scale
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
            else:
                break

        # Try moving circles slightly to allow expansion
        # This is complex, so we just do a quick force relaxation with high k_rep
        # to push them apart, then expand
        # Actually, the simulation already did this.
        # Let's just do a few iterations of "expand and fix"
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                # Try to expand circle i
                # Max radius limited by walls and neighbors
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-6:
                    # Try to set to r_max
                    # But this might make it invalid if it was touching?
                    # No, if d - r[j] is the limit, then d = r[i]_new + r[j], so it's touching.
                    # Validity check allows touching (d >= r_sum - tol)
                    r[i] = r_max
                    improved = True

            if not improved:
                break

        # Verify and update best
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w = best_result
    return np.column_stack((x, y, r))
