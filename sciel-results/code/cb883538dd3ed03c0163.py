import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    def validate(x, y, r, w, h, tol=1e-6):
        """Check all constraints strictly."""
        if w + h > 2.0 + 1e-8 or w < 0.05 or h < 0.05:
            return False
        for i in range(N):
            if r[i] < -tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def objective(params, w, h):
        """Objective to minimize: negative sum of radii + penalty for violations."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Penalty for wall violations
        penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                penalty += 100 * (r[i] - x[i])**2
            if x[i] > w - r[i]:
                penalty += 100 * (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                penalty += 100 * (r[i] - y[i])**2
            if y[i] > h - r[i]:
                penalty += 100 * (y[i] - (h - r[i]))**2
            if r[i] < 0:
                penalty += 100 * r[i]**2
        
        # Penalty for circle overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    penalty += 100 * (min_d - d)**2
        
        return -np.sum(r) + penalty

    def physics_simulate(x0, y0, r0, w0, h0, steps=8000, dt=0.0015, damping=0.97):
        """
        Physics-based relaxation with soft constraints.
        """
        x = x0.copy().astype(np.float64)
        y = y0.copy().astype(np.float64)
        r = r0.copy().astype(np.float64)
        w = w0
        h = h0

        vx = np.zeros(N)
        vy = np.zeros(N)
        vr = np.zeros(N)

        best_local = -np.inf
        best_local_state = None

        for step in range(steps):
            if time.time() > deadline - 3:
                break

            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)

            # Inter-circle repulsion (Lennard-Jones-like)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-10:
                        dx = 0.01 * (np.random.rand() - 0.5)
                        dy = 0.01 * (np.random.rand() - 0.5)
                        d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d * 1.2:
                        # Repulsive force
                        overlap = max(0, min_d - d)
                        force = 50.0 * overlap / max(d, 1e-6)
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
                        # Reduce radii when overlapping
                        fr[i] -= 2.0 * overlap
                        fr[j] -= 2.0 * overlap

            # Wall constraints and radius growth pressure
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += 50.0 * (r[i] - x[i])
                # Right wall
                if x[i] > w - r[i]:
                    fx[i] -= 50.0 * (x[i] - (w - r[i]))
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += 50.0 * (r[i] - y[i])
                # Top wall
                if y[i] > h - r[i]:
                    fy[i] -= 50.0 * (y[i] - (h - r[i]))

                # Compute max possible radius for circle i
                max_r_wall = min(x[i], w - x[i], y[i], h - y[i])
                max_r = max_r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-10:
                        max_r = min(max_r, (d - r[j]))
                max_r = max(0.001, max_r)

                # Grow radius toward max possible
                fr[i] += 1.0 * (max_r - r[i])

            # Update velocities
            vx = damping * vx + fx * dt
            vy = damping * vy + fy * dt
            vr = damping * vr + fr * dt

            # Update positions
            x += vx * dt
            y += vy * dt
            r += vr * dt

            # Enforce hard constraints
            for i in range(N):
                r[i] = max(0.001, r[i])
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

            val = np.sum(r)
            if val > best_local:
                best_local = val
                best_local_state = (x.copy(), y.copy(), r.copy(), w, h)

        return best_local_state

    def grid_init(w, h, n_cols, n_rows):
        """Create a grid initialization."""
        xs = []
        ys = []
        rs = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(xs) >= N:
                    break
                x = (j + 0.5) * w / n_cols
                y = (i + 0.5) * h / n_rows
                r = min(w / n_cols, h / n_rows) / 2 * 0.7
                xs.append(x)
                ys.append(y)
                rs.append(r)
        while len(xs) < N:
            xs.append(w / 2)
            ys.append(h / 2)
            rs.append(0.03)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def random_init(w, h):
        """Random initialization."""
        xs = []
        ys = []
        rs = []
        for _ in range(N):
            for _ in range(200):
                r = np.random.uniform(0.02, min(w, h) / 3)
                x = np.random.uniform(r, w - r)
                y = np.random.uniform(r, h - r)
                ok = True
                for i in range(len(xs)):
                    d = np.sqrt((x - xs[i])**2 + (y - ys[i])**2)
                    if d < r + rs[i] + 0.02:
                        ok = False
                        break
                if ok:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
                    break
            if len(xs) <= len(rs):  # circle was placed
                pass
            else:
                xs.append(w / 2 + np.random.uniform(-0.1, 0.1))
                ys.append(h / 2 + np.random.uniform(-0.1, 0.1))
                rs.append(0.02)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def scipy_optimize(x0, y0, r0, w, h, maxiter=500):
        """Use scipy minimize to refine."""
        params0 = np.concatenate([x0, y0, r0])
        
        # Define bounds
        bounds = []
        for i in range(N):
            bounds.append((0.001, w - 0.001))  # x
        for i in range(N):
            bounds.append((0.001, h - 0.001))  # y
        for i in range(N):
            bounds.append((0.001, min(w, h) / 2))  # r
        
        try:
            result = minimize(
                lambda p: objective(p, w, h),
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < objective(params0, w, h):
                x = result.x[:N]
                y = result.x[N:2*N]
                r = result.x[2*N:3*N]
                return x, y, r
        except:
            pass
        return None

    best_val = -np.inf
    best_out = None

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 0.8, 0.66, 0.5, 1.8]

    for aspect in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = aspect / (1 + aspect) * 2.0
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            continue

        # Try grid inits
        grid_configs = [(5, 5), (4, 6), (3, 7), (6, 4), (7, 3), (5, 4), (4, 5), (3, 4)]
        for (nc, nr) in grid_configs[:4]:
            if time.time() > deadline - 10:
                break
            x0, y0, r0 = grid_init(w, h, nc, nr)
            
            # Physics simulation
            result = physics_simulate(x0, y0, r0, w, h, steps=5000)
            if result:
                x, y, r, w_r, h_r = result
                if validate(x, y, r, w_r, h_r):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_out = np.column_stack([x, y, r])

            # Scipy refinement
            if time.time() < deadline - 15:
                ref = scipy_optimize(x, y, r, w, h, maxiter=300)
                if ref:
                    x2, y2, r2 = ref
                    if validate(x2, y2, r2, w, h):
                        val = np.sum(r2)
                        if val > best_val:
                            best_val = val
                            best_out = np.column_stack([x2, y2, r2])

        # Random inits
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            x0, y0, r0 = random_init(w, h)
            result = physics_simulate(x0, y0, r0, w, h, steps=5000)
            if result:
                x, y, r, w_r, h_r = result
                if validate(x, y, r, w_r, h_r):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_out = np.column_stack([x, y, r])

                # Scipy refinement
                if time.time() < deadline - 15:
                    ref = scipy_optimize(x, y, r, w, h, maxiter=300)
                    if ref:
                        x2, y2, r2 = ref
                        if validate(x2, y2, r2, w, h):
                            val = np.sum(r2)
                            if val > best_val:
                                best_val = val
                                best_out = np.column_stack([x2, y2, r2])

    # Final refinement on best result
    if best_out is not None and time.time() < deadline - 20:
        x = best_out[:, 0].copy()
        y = best_out[:, 1].copy()
        r = best_out[:, 2].copy()
        
        # Estimate w, h from bounding box
        w_est = np.max(x) + np.max(r)
        h_est = np.max(y) + np.max(r)
        scale = 4.0 / (2 * (w_est + h_est))
        w_final = w_est * scale
        h_final = h_est * scale
        x = x * scale
        y = y * scale
        r = r * scale

        # More aggressive physics
        result = physics_simulate(x, y, r, w_final, h_final, steps=10000, dt=0.001, damping=0.98)
        if result:
            x2, y2, r2, w2, h2 = result
            if validate(x2, y2, r2, w2, h2):
                val = np.sum(r2)
                if val > best_val:
                    best_val = val
                    best_out = np.column_stack([x2, y2, r2])

        # Scipy polish
        if time.time() < deadline - 10:
            ref = scipy_optimize(x2, y2, r2, w2, h2, maxiter=500)
            if ref:
                x3, y3, r3 = ref
                if validate(x3, y3, r3, w2, h2):
                    val = np.sum(r3)
                    if val > best_val:
                        best_val = val
                        best_out = np.column_stack([x3, y3, r3])

    # Fallback
    if best_out is None:
        out = np.zeros((N, 3))
        side = 5
        for i in range(N):
            row = i // side
            col = i % side
            out[i, 0] = (col + 0.5) / side
            out[i, 1] = (row + 0.5) / side
            out[i, 2] = 0.5 / side - 0.001
        best_out = out

    return best_out
