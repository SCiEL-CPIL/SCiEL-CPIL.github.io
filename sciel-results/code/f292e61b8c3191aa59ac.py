<<<<<<< SEARCH
def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = np.sum(r)
    best_xyz = xyz.copy()
    last_improvement = 0.0

    dt = 0.005
    current_grow = grow_rate

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        if iteration % 100 == 0:
            if (time.time() - start_time) - last_improvement > 2.0:
                current_grow *= 0.7
            else:
                current_grow = grow_rate

        r *= (1.0 + current_grow)

        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        fx = np.zeros(N)
        fy = np.zeros(N)

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        min_dist = r[idx_i] + r[idx_j]
        overlap = min_dist - dist
        active = overlap > 0

        if np.any(active):
            force = np.zeros(len(active))
            force[active] = overlap[active] * 80.0
            inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)

            np.add.at(fx, idx_i, force * dx * inv_dist)
            np.add.at(fx, idx_j, -force * dx * inv_dist)
            np.add.at(fy, idx_i, force * dy * inv_dist)
            np.add.at(fy, idx_j, -force * dy * inv_dist)

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0

        x += dt * fx
        y += dt * fy

        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
                last_improvement = time.time() - start_time

    return best_xyz
=======
def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = np.sum(r)
    best_xyz = xyz.copy()
    last_improvement = 0.0

    dt = 0.005
    current_grow = grow_rate

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        if iteration % 100 == 0:
            if (time.time() - start_time) - last_improvement > 2.0:
                current_grow *= 0.7
            else:
                current_grow = grow_rate

        r *= (1.0 + current_grow)

        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        fx = np.zeros(N)
        fy = np.zeros(N)

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        min_dist = r[idx_i] + r[idx_j]
        overlap = min_dist - dist
        active = overlap > 0

        if np.any(active):
            force = np.zeros(len(active))
            force[active] = overlap[active] * 80.0
            inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)

            np.add.at(fx, idx_i, force * dx * inv_dist)
            np.add.at(fx, idx_j, -force * dx * inv_dist)
            np.add.at(fy, idx_i, force * dy * inv_dist)
            np.add.at(fy, idx_j, -force * dy * inv_dist)

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0

        x += dt * fx
        y += dt * fy

        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
                last_improvement = time.time() - start_time

    return best_xyz


def _lm_packing(params, time_budget=5.0, smooth_eps=1e-3):
    """
    Levenberg-Marquardt solver for circle packing.
    Minimizes sum of squared softplus violations of overlap and boundary constraints
    while maximizing sum of radii.
    """
    start_time = time.time()
    n = N
    params = params.copy().astype(float)
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    # Initial lambda
    lam = 1.0
    max_iter = 200
    ftol = 1e-14
    gtol = 1e-12

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        x = params[0::3].copy()
        y = params[1::3].copy()
        r = params[2::3].copy()

        # Compute residuals and Jacobian
        residuals = []
        J_rows = []

        # Overlap constraints
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        overlap = r[idx_i] + r[idx_j] - dist

        for k in range(len(overlap)):
            ov = overlap[k]
            if ov > -smooth_eps:
                # softplus(ov)
                sp = np.logaddexp(0, ov / smooth_eps) * smooth_eps
                sg = 1.0 / (1.0 + np.exp(-ov / smooth_eps))
                residuals.append(sp)

                # Gradient of softplus wrt x_i, y_i, r_i, x_j, y_j, r_j
                row = np.zeros(3 * n)
                inv_dist = 1.0 / dist[k]
                factor = 2.0 * sp * sg / smooth_eps

                # d(overlap)/dx_i = -dx/dist, d(overlap)/dy_i = -dy/dist
                # d(overlap)/dr_i = 1, d(overlap)/dx_j = dx/dist, d(overlap)/dy_j = dy/dist, d(overlap)/dr_j = 1
                row[idx_i * 3 + 0] = -factor * dx[k] * inv_dist
                row[idx_i * 3 + 1] = -factor * dy[k] * inv_dist
                row[idx_i * 3 + 2] = factor
                row[idx_j * 3 + 0] = factor * dx[k] * inv_dist
                row[idx_j * 3 + 1] = factor * dy[k] * inv_dist
                row[idx_j * 3 + 2] = factor
                J_rows.append(row)

        # Boundary constraints
        for i in range(n):
            # x - r < 0
            val = r[i] - x[i]
            if val > -smooth_eps:
                sp = np.logaddexp(0, val / smooth_eps) * smooth_eps
                sg = 1.0 / (1.0 + np.exp(-val / smooth_eps))
                residuals.append(sp)
                row = np.zeros(3 * n)
                factor = 2.0 * sp * sg / smooth_eps
                row[i * 3 + 0] = -factor
                row[i * 3 + 2] = factor
                J_rows.append(row)

            # x + r > 1
            val = x[i] + r[i] - 1.0
            if val > -smooth_eps:
                sp = np.logaddexp(0, val / smooth_eps) * smooth_eps
                sg = 1.0 / (1.0 + np.exp(-val / smooth_eps))
                residuals.append(sp)
                row = np.zeros(3 * n)
                factor = 2.0 * sp * sg / smooth_eps
                row[i * 3 + 0] = factor
                row[i * 3 + 2] = factor
                J_rows.append(row)

            # y - r < 0
            val = r[i] - y[i]
            if val > -smooth_eps:
                sp = np.logaddexp(0, val / smooth_eps) * smooth_eps
                sg = 1.0 / (1.0 + np.exp(-val / smooth_eps))
                residuals.append(sp)
                row = np.zeros(3 * n)
                factor = 2.0 * sp * sg / smooth_eps
                row[i * 3 + 1] = -factor
                row[i * 3 + 2] = factor
                J_rows.append(row)

            # y + r > 1
            val = y[i] + r[i] - 1.0
            if val > -smooth_eps:
                sp = np.logaddexp(0, val / smooth_eps) * smooth_eps
                sg = 1.0 / (1.0 + np.exp(-val / smooth_eps))
                residuals.append(sp)
                row = np.zeros(3 * n)
                factor = 2.0 * sp * sg / smooth_eps
                row[i * 3 + 1] = factor
                row[i * 3 + 2] = factor
                J_rows.append(row)

        if len(residuals) == 0:
            # No violations, try to grow radii
            # Add a small negative residual for each radius to encourage growth
            for i in range(n):
                residuals.append(-r[i] * 0.01)
                row = np.zeros(3 * n)
                row[i * 3 + 2] = -0.01
                J_rows.append(row)

        if len(residuals) == 0:
            break

        res = np.array(residuals)
        J = np.array(J_rows)

        # Compute J^T J and J^T r
        JtJ = J.T @ J
        Jtr = J.T @ res

        # Add gradient of -sum(r) to Jtr
        Jtr[2::3] += -1.0

        # Solve (J^T J + lam * I) delta = -J^T r
        for _ in range(10):
            try:
                delta = np.linalg.solve(JtJ + lam * np.eye(3 * n), -Jtr)
                break
            except np.linalg.LinAlgError:
                lam *= 10
                if lam > 1e10:
                    delta = np.zeros(3 * n)
                    break

        new_params = params + delta

        # Clip to valid range
        new_params[0::3] = np.clip(new_params[0::3], 0, 1)
        new_params[1::3] = np.clip(new_params[1::3], 0, 1)
        new_params[2::3] = np.clip(new_params[2::3], 1e-6, 0.5)

        # Evaluate new objective
        new_xyz = new_params.reshape(n, 3)
        if _validate(new_xyz):
            new_score = np.sum(new_xyz[:, 2])
            if new_score > best_score:
                best_score = new_score
                best_params = new_params
            # Accept step
            params = new_params
            lam = max(lam * 0.5, 1e-8)
        else:
            # Reject step, increase lambda
            lam *= 10
            if lam > 1e10:
                break

        # Check convergence
        if np.linalg.norm(delta) < ftol:
            break

    return best_params
>>>>>>> REPLACE

