<<<<<<< SEARCH
    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        """SLSQP with smooth penalty objective and explicit bounds."""
        def objective(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            w = z[3*N]
            h = 2.0 - w

            if w < 0.3 or h < 0.3:
                return 1e8

            # Smooth objective: negative sum of radii + smooth overlap/wall penalties
            obj = -np.sum(r)

            # Smooth overlap penalty using soft-plus
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    # Smooth max(ov, 0) using log-sum-exp style
                    if ov > 0.01:
                        obj += 500.0 * ov * ov
                    elif ov > -0.01:
                        obj += 500.0 * ov * ov  # quadratic near zero is already smooth

            # Smooth wall penalties
            for i in range(N):
                for bound in [(r[i] - x[i]), (x[i] + r[i] - w), (r[i] - y[i]), (y[i] + r[i] - h)]:
                    if bound > 0.005:
                        obj += 500.0 * bound * bound
                    elif bound > -0.005:
                        obj += 500.0 * bound * bound

            return obj

        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])

        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None

        maxiter = int(min(500, max(100, remaining * 25)))

        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, 0.45))  # x
        for i in range(N):
            h_init = 2.0 - w_init
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))  # y
        for i in range(N):
            bounds.append((0.01, 0.45))  # r
        bounds.append((0.35, 1.65))  # w

        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]

            if np.any(r_opt < 0.005):
                return None

            # Validate
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt

            # Try to fix small violations
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt

        except Exception:
            pass

        return None

    def relax_positions(x, y, r, w, max_iter=300, ftol=1e-14):
        """Relax positions via L-BFGS-B to minimize overlap and wall penalties."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]

        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w, max_rounds=5):
        """Iteratively grow radii and relax positions."""
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                for _ in range(50):
                    r_new = r.copy()
                    r_new[i] *= 1.003
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=40, ftol=1e-12)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        """Nelder-Mead polish on positions only."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w
=======
    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        """SLSQP with smooth penalty objective and explicit bounds."""
        def objective(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            w = z[3*N]
            h = 2.0 - w

            if w < 0.3 or h < 0.3:
                return 1e8

            obj = -np.sum(r)

            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        obj += 500.0 * ov * ov

            for i in range(N):
                for bound in [(r[i] - x[i]), (x[i] + r[i] - w), (r[i] - y[i]), (y[i] + r[i] - h)]:
                    if bound > 0:
                        obj += 500.0 * bound * bound

            return obj

        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])

        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None

        maxiter = int(min(300, max(80, remaining * 20)))

        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.45))
        for i in range(N):
            h_init = 2.0 - w_init
            bounds.append((0.01, max(0.01, h_init - 0.01)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))

        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]

            if np.any(r_opt < 0.005):
                return None

            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt

        except Exception:
            pass

        return None

    def try_grow(x, y, r, w, max_rounds=3):
        """Iteratively grow radii."""
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                r_new = r.copy()
                r_new[i] *= 1.005
                # Simple position adjustment
                x_t, y_t = x.copy(), y.copy()
                for j in range(N):
                    if j != i:
                        dx = x_t[i] - x_t[j]
                        dy = y_t[i] - y_t[j]
                        d = np.sqrt(dx*dx + dy*dy) + 1e-12
                        min_d = r_new[i] + r_new[j]
                        if d < min_d:
                            push = (min_d - d) / 2
                            ux, uy = dx/d, dy/d
                            x_t[i] += ux * push
                            y_t[i] += uy * push
                            x_t[j] -= ux * push
                            y_t[j] -= uy * push
                
                # Clip
                for k in range(N):
                    x_t[k] = np.clip(x_t[k], r_new[k], w - r_new[k])
                    y_t[k] = np.clip(y_t[k], r_new[k], h - r_new[k])
                
                if is_valid(x_t, y_t, r_new, w):
                    x, y, r = x_t, y_t, r_new
                    improved = True
            
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        """Simple polish via coordinate descent."""
        h = 2.0 - w
        x_t, y_t = x.copy(), y.copy()
        
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            for i in range(N):
                # Try small perturbations
                for dx in [-0.002, -0.001, 0.001, 0.002]:
                    for dy in [-0.002, -0.001, 0.001, 0.002]:
                        x_try = x_t.copy()
                        y_try = y_t.copy()
                        x_try[i] += dx
                        y_try[i] += dy
                        x_try[i] = np.clip(x_try[i], r[i], w - r[i])
                        y_try[i] = np.clip(y_try[i], r[i], h - r[i])
                        if is_valid(x_try, y_try, r, w):
                            # Check if it improves minimum separation
                            x_t, y_t = x_try, y_try
                            break
                    else:
                        continue
                    break
        
        if is_valid(x_t, y_t, r, w):
            return x_t, y_t, r, w
        return x, y, r, w
>>>>>>> REPLACE

Now let me fix the fallback at the end:

