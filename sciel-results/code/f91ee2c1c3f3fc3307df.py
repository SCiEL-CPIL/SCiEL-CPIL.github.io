"""Circle packing in perimeter-4 rectangle: force-directed pressure simulation."""
import numpy as np
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def simulate(w0, h0, time_budget, seed=0):
        """Run force-directed pressure simulation for a given rectangle."""
        rng = np.random.default_rng(seed)
        
        # Grid-based initial placement for better starting positions
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        cx = w0 / cols
        cy = h0 / rows
        x = np.empty(N)
        y = np.empty(N)
        r = np.empty(N)
        for i in range(N):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) * cx + rng.normal(0, 0.01)
            y[i] = (row + 0.5) * cy + rng.normal(0, 0.01)
            r[i] = min(cx, cy) * 0.45 + rng.uniform(0, 0.01)
        
        # Clip to valid range
        x = np.clip(x, 0.05, w0 - 0.05)
        y = np.clip(y, 0.05, h0 - 0.05)

        vx = np.zeros(N)
        vy = np.zeros(N)
        dt = 0.003
        damping = 0.85
        iu = np.triu_indices(N, k=1)

        def forces(x, y, r, inflate):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall repulsion - stronger to keep circles inside
            fx -= np.maximum(r - x, 0.0) * 80.0
            fx += np.maximum(x + r - w0, 0.0) * 80.0
            fy -= np.maximum(r - y, 0.0) * 80.0
            fy += np.maximum(y + r - h0, 0.0) * 80.0
            # Gentle center attraction to prevent drift
            fx += (w0 * 0.5 - x) * 0.3
            fy += (h0 * 0.5 - y) * 0.3
            # Pairwise repulsion - proportional to overlap
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx * dx + dy * dy
            rs = r[:, None] + r[None, :]
            dp = np.sqrt(np.maximum(d2[iu], 1e-12))
            overlap = np.maximum(rs[iu] - dp, 0.0)
            fp = overlap * 30.0
            ux = dx[iu] / dp
            uy = dy[iu] / dp
            for k in range(len(iu[0])):
                a = iu[0][k]; b = iu[1][k]
                fx[a] += ux[k] * fp[k]; fx[b] -= ux[k] * fp[k]
                fy[a] += uy[k] * fp[k]; fy[b] -= uy[k] * fp[k]
            return fx, fy

        def step(x, y, r, vx, vy, dt, damping):
            fx, fy = forces(x, y, r, True)
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x = x + vx * dt
            y = y + vy * dt
            x = np.clip(x, r, w0 - r)
            y = np.clip(y, r, h0 - r)
            return x, y, vx, vy

        # Phase A: settle positions to resolve overlaps
        for it in range(800):
            x, y, vx, vy = step(x, y, r, vx, vy, dt, damping)
            if time.time() > start_time + time_budget * 0.3:
                break

        # Phase B: inflate radii gradually with multiple restarts
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum = float(np.sum(r))
        
        base = r.copy()
        for attempt in range(3):
            r = base.copy()
            x_curr, y_curr = x.copy(), y.copy()
            vx_curr, vy_curr = vx.copy(), vy.copy()
            
            for phase in range(30):
                target = 1.0 + 0.06 * phase
                for sub in range(100):
                    if time.time() > start_time + time_budget:
                        break
                    r = base * (1.0 + (target - 1.0) * (sub + 1) / 100.0)
                    x_curr, y_curr, vx_curr, vy_curr = step(x_curr, y_curr, r, vx_curr, vy_curr, dt, damping)
                    if sub == 99:
                        cur = float(np.sum(r))
                        if cur > best_sum:
                            best_sum = cur
                            best_x, best_y, best_r = x_curr.copy(), y_curr.copy(), r.copy()
            
            # Add random perturbation for next attempt
            if attempt < 2:
                x_curr += rng.normal(0, 0.02, N)
                y_curr += rng.normal(0, 0.02, N)
                x_curr = np.clip(x_curr, 0.05, w0 - 0.05)
                y_curr = np.clip(y_curr, 0.05, h0 - 0.05)
                vx_curr = np.zeros(N)
                vy_curr = np.zeros(N)
        
        return best_x, best_y, best_r, w0, h0

    def expand_uniform(x, y, r, w, h, time_budget):
        """Binary-search uniform radius scale that stays valid."""
        lo = 1.0
        hi = 1.30
        # find a feasible hi
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if check_validity(x, y, r * mid, w, h)[0]:
                lo = mid
            else:
                hi = mid
            if time.time() > start_time + time_budget:
                break
        return r * lo

    def relocate_polish(x, y, r, w, h, time_budget):
        """Nudge each circle to its max-radius position, then re-expand."""
        x = x.copy(); y = y.copy(); r = r.copy()
        rng = np.random.default_rng(42)
        for rnd in range(12):
            if time.time() > start_time + time_budget:
                break
            improved = False
            for i in range(N):
                # max radius at current position
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                rmax = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    rmax = min(rmax, d - r[j])
                rmax = max(0.0, rmax)
                if rmax > r[i] + 1e-6:
                    r[i] = rmax
                    improved = True
                # try small moves to open more space - more directions
                if r[i] < rmax - 1e-5:
                    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    for angle in angles:
                        step_size = 0.015
                        nx = x[i] + step_size * np.cos(angle)
                        ny = y[i] + step_size * np.sin(angle)
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        ok = True
                        for j in range(N):
                            if i == j:
                                continue
                            d2 = (nx - x[j]) ** 2 + (ny - y[j]) ** 2
                            if d2 < (r[i] + r[j]) ** 2 - 1e-9:
                                ok = False
                                break
                        if ok:
                            x[i] = nx
                            y[i] = ny
                            improved = True
                            break
                    # Also try random perturbations
                    if not improved:
                        for _ in range(5):
                            dx = rng.normal(0, 0.02)
                            dy = rng.normal(0, 0.02)
                            nx = x[i] + dx
                            ny = y[i] + dy
                            if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                                continue
                            ok = True
                            for j in range(N):
                                if i == j:
                                    continue
                                d2 = (nx - x[j]) ** 2 + (ny - y[j]) ** 2
                                if d2 < (r[i] + r[j]) ** 2 - 1e-9:
                                    ok = False
                                    break
                            if ok:
                                x[i] = nx
                                y[i] = ny
                                improved = True
                                break
            if not improved:
                break
        return x, y, r

    # Sweep a range of aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 0.9, 1.1, 0.7, 1.4, 0.65, 1.25]
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.35 or w0 < 0.35:
            continue
        remaining = deadline - (time.time() - start_time)
        if remaining < 3.0:
            break
        budget = min(5.0, remaining - 1.5)
        try:
            x, y, r, w, h = simulate(w0, h0, budget, seed=1000 + k)
            # uniform expansion
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 2)
            # relocate polish
            x, y, r = relocate_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 2)
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
        except Exception:
            continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    # Final safety shrink
    for _ in range(8):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
